from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from research_agent.agent.config import LLM_CONFIG
from research_agent.tools.search_paper import search_papers
from typing import List, Dict
import json
from statistics import mean

# ============================================================================
# STEP 1: Create Agents
# ============================================================================

paper_agent = AssistantAgent(
    name="research_agent",
    llm_config=LLM_CONFIG,
    system_message=(
        "You are a research paper search assistant inside a multi-agent system.\n"
        "You only talk to other agents, never the human.\n"
        "Workflow:\n"
        "1) Read USER_REQUEST carefully.\n"
        "2) ALWAYS use the search_papers tool to find papers, Always Always Always. Do not invent papers. always include links to the paper\n"
        "3) Format your first answer as 'FIRST DRAFT:' for the internal critic to review.\n"
        "4) You must output your parameters chosen for the search_paper tool call as part of your DRAFT \n"
        "5) YOU absolutely have to state how many papaers you found in total in the tool call \n"
        "6) If internal_critic has any critique, you must adhere to it, but you can ask for clarification if you disagree \n"
        "7) If internal_critic says OK, send 'FINAL_ANSWER:' including 'TERMINATE'.\n"

    ),
)

paper_agent.register_for_llm(
    name="search_papers",
    description=(
        "Search for academic research papers. Returns real papers from Semantic Scholar.\n"
        "Parameters:\n"
        "- topic (str, required)\n"
        "- year (int, optional)\n"
        "- year_filter (str, optional): 'exact', 'before', 'after'\n"
        "- min_citations (int, optional)\n"
        "- limit (int, optional, default 10)"
    )
)(search_papers)

internal_critic = AssistantAgent(
    name="internal_critic",
    llm_config=LLM_CONFIG,
    system_message=(
        "You are an internal critic reviewing research_agent's DRAFTs.\n"
        "Evaluate completeness, accuracy, tool usage, and clarity. \n"
        "If a paper is very relevant but outside year scope you are allowed to be lenient "
        "Put high value in proper paramater usage and make sure all links are relevant \n"
        "Respond ONLY with:\n"
        "- OK: <short justification>\n"
        "- CRITIQUE: <issue + smallest fix>\n"
        "Do NOT produce a final answer. \n"
        "Never use code blocks ´´´´´´"
    ),
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda m: "TERMINATE" in m.get("content", ""),
)
user_proxy.register_for_execution(name="search_papers")(search_papers)

# ============================================================================
# STEP 2: Create GroupChat
# ============================================================================

def make_groupchat() -> GroupChatManager:
    group = GroupChat(
        agents=[user_proxy, paper_agent, internal_critic],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(groupchat=group, llm_config=LLM_CONFIG)
    return manager

# ============================================================================
# STEP 3: Run User Request Through GroupChat
# ============================================================================

def run_with_internal_critic(user_request: str) -> Dict:
    manager = make_groupchat()

    init_message = (
        f"USER_REQUEST:\n{user_request}\n\n"
        "Workflow for agents:\n"
        "- paper_agent: read USER_REQUEST and propose an answer as 'FIRST DRAFT: ...'.\n"
        "- internal_critic: when you see a any type of DRAFT, respond with 'OK:' or 'CRITIQUE:'\n"
        "- paper_agent: if you get CRITIQUE, revise and send a new 'NEW DRAFT:'\n"
        "- When and ONLY when the internal_critic is satisfied, indicated by saying OK, paper_agent sends \n"
        "'FINAL_ANSWER: ...' and also includes 'TERMINATE' in the same message.\n"
        "The human will only see the FINAL_ANSWER.\n"
    )

    # Send raw string message to manager
    final = user_proxy.initiate_chat(manager, message=init_message, role_override="assistant")

    trace = list(manager.groupchat.messages)

    # Extract FINAL_ANSWER from paper_agent
    final_answer = None
    for msg in reversed(trace):
        if msg.get("name") == "research_agent" and isinstance(msg.get("content"), str):
            content = msg["content"]
            if "FINAL_ANSWER:" in content:
                final_answer = content
                break

    return {
        "final_answer": final_answer or str(final),
        "trace": trace,
    }

# ============================================================================
# STEP 4: External Judge
# ============================================================================

def create_judge_agent() -> AssistantAgent:
    return AssistantAgent(
        name="judge_agent",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are a strict evaluation agent. You will be given two items:\n"
            "1) PROMPT – the user's original request.\n"
            "2) ANSWER – the research agent's final response.\n\n"
            "Your task is to evaluate the ANSWER objectively based on the PROMPT, and return STRICT JSON only.\n"
            "Fields to return:\n"
            "- completeness (int, 1-5): how fully the ANSWER addresses the PROMPT\n"
            "- accuracy (int, 1-5): factual correctness, including proper citations and paper details\n"
            "- quality (int, 1-5): clarity, structure, and readability of the ANSWER\n"
            "- robustness (int, 1-5): proper tool usage, handling of edge cases, and consistency\n"
            "- feedback (string): concise explanation of your scores or issues\n\n"
            "Requirements:\n"
            "- Do not output anything except valid JSON.\n"
            "- Use integers 1-5 for scoring, never decimals.\n"
            "- Be strict: if information is missing, inaccurate, or misformatted, lower the score.\n"
            "- Do not make up papers or citations.\n"
            "- Avoid extra commentary or apologies.\n"
            "- Example output format:\n"
            '{"completeness": 4, "accuracy": 5, "quality": 4, "robustness": 3, "feedback": "Some papers missing links."}'
        ),
    )

def llm_judge_score(user_prompt: str, final_answer: str) -> Dict:
    judge = create_judge_agent()
    response = judge.generate_reply(
        messages=[{
            "role": "user",
            "content": f'PROMPT: "{user_prompt}"\nANSWER: "{final_answer}"\nReturn JSON only.'
        }]
    )
    content = response.get("content") if isinstance(response, dict) else str(response)

    try:
        return json.loads(content.strip())
    except Exception as e:
        return {
            "completeness": 0,
            "accuracy": 0,
            "quality": 0,
            "robustness": 0,
            "feedback": f"Judge error: {e}"
        }

# ============================================================================
# STEP 5: Test Prompts
# ============================================================================

TEST_PROMPTS: List[str] = [
    "Find papers about machine learning from 2020 with at least 500 citations",
    "Search for deep learning papers published after 2018",
    "I need papers about neural networks",
    "Find research on reinforcement learning with 1000+ citations",
    "Recent NLP papers from 2022 with at least 200 citations",
]

# ============================================================================
# STEP 6: Evaluation & Summary
# ============================================================================

def evaluate_prompt(prompt: str) -> Dict:
    internal_result = run_with_internal_critic(prompt)
    final_answer = internal_result["final_answer"]
    judge_scores = llm_judge_score(prompt, final_answer)

    return {
        "prompt": prompt,
        "final_answer": final_answer,
        "judge_scores": judge_scores,
        "rounds": len(internal_result["trace"]),
    }

def run_batch_evaluation(prompts: List[str] = None) -> List[Dict]:
    if prompts is None:
        prompts = TEST_PROMPTS

    results = []
    for p in prompts:
        try:
            result = evaluate_prompt(p)
            results.append(result)
        except Exception as e:
            results.append({
                "prompt": p,
                "final_answer": f"ERROR: {e}",
                "judge_scores": {"completeness": 0, "accuracy": 0, "quality": 0, "robustness": 0, "feedback": str(e)},
                "rounds": 0
            })
    return results

def print_summary(results: List[Dict]):
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['prompt']}")
        print(f"   Scores: {r['judge_scores']}")

    def avg(key: str) -> float:
        valid = [r["judge_scores"][key] for r in results if isinstance(r["judge_scores"].get(key), (int, float))]
        return mean(valid) if valid else 0.0

    print("\n" + "="*80)
    print(f"Avg Completeness: {avg('completeness'):.2f}/5")
    print(f"Avg Accuracy:     {avg('accuracy'):.2f}/5")
    print(f"Avg Quality:      {avg('quality'):.2f}/5")
    print(f"Avg Robustness:   {avg('robustness'):.2f}/5")
    print("="*80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Research Agent Evaluation WITH GroupChat + Internal Critic\n")
    results = run_batch_evaluation()
    print_summary(results)

    with open("tool_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to tool_evaluation_results.json")
