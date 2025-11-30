"""
Agent Evaluation System for Research Paper Agent

This module implements both internal (critic in the loop) and external 
(LLM-as-judge) evaluation for the research paper search agent.
"""

from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from research_agent.agent.config import LLM_CONFIG
from research_agent.tools.search_paper import search_papers
from typing import Dict, List
import json
from statistics import mean


# ============================================================================
# STEP 1: Define Agents for Internal Evaluation (GroupChat)
# ============================================================================

def create_research_agent_for_groupchat() -> AssistantAgent:
    """Research agent that works with internal critic in GroupChat"""
    agent = AssistantAgent(
        name="research_agent",
        llm_config=LLM_CONFIG,
        system_message=(
"You are a research paper search assistant inside a multi-agent system.\n"
            "You talk only to other agents, not the human.\n\n"
            "**STRICT Workflow - Tool Use IS Mandatory for Draft:**\n"
            "1) When you receive the original USER_REQUEST, you **MUST FIRST** use the "
            "`search_papers` tool with the appropriate parameters to gather data. "
            "You **CANNOT** produce a 'DRAFT:' answer without a successful `search_papers` "
            "tool call and its results in the conversation history. This step is "
            "NON-NEGOTIABLE for your first response.\n"
            "2) After receiving the search results, produce a draft answer "
            "prefixed with 'DRAFT:'.\n"
            "3) Wait for the internal_critic to respond with 'OK:' or 'CRITIQUE:'.\n"
            "4) If you receive CRITIQUE, revise your search (call the tool again if needed) "
            "or revise your answer and send a new 'DRAFT:' that addresses the critique.\n"
            "5) When the critic responds with OK, send the final answer prefixed with "
            "'FINAL_ANSWER:' and include the token 'TERMINATE' in the same message.\n\n"
            "Guidelines:\n"
            "- **PRIORITY 1:** Your very first action, when given the USER_REQUEST, "
            "must be a call to the `search_papers` tool.\n"
            "- Use search_papers tool to find papers matching the criteria\n"
            "- Present results clearly with title, authors, year, citations, and abstract/summary\n"
            "- If criteria are ambiguous, explain your interpretation *after* the initial tool call\n"
            "- If no papers match, explain why in your 'DRAFT:' and suggest alternatives\n"
        ),
    )
    
    # Register the search tool
    agent.register_for_llm(
        name="search_papers",
        description=(
            "Search for academic research papers. "
            "Parameters: topic (required), year (optional), min_citations (optional), "
            "year_filter ('exact', 'before', 'after'), max_results (default 10)"
        )
    )(search_papers)
    
    return agent


def create_internal_critic() -> AssistantAgent:
    """Internal critic that reviews research agent's drafts"""
    return AssistantAgent(
        name="internal_critic",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are an internal critic reviewing the research_agent's draft answers.\n"
            "You only see the USER_REQUEST and the research_agent's messages.\n\n"
            "Evaluation criteria:\n"
            "- completeness: Did the agent address ALL parts of the request? "
            "(topic, year filter, citation count)\n"
            "- accuracy: Are the search parameters correct? Does year filter match request?\n"
            "- quality: Are results presented clearly with all relevant info?\n"
            "- robustness: If request is ambiguous, did agent handle it sensibly?\n\n"
            "Rules:\n"
            "- If the latest message from research_agent starts with 'DRAFT:' and is "
            "acceptable, respond with:\n"
            "  OK: <short justification>\n"
            "- If there are issues, respond with:\n"
            "  CRITIQUE: <what is wrong + how to fix it>\n"
            "- Do NOT search for papers yourself; only judge the agent's work.\n"
        ),
    )


def create_user_proxy_for_groupchat() -> UserProxyAgent:
    """User proxy to drive the GroupChat"""
    user_proxy = UserProxyAgent(
        name="user_proxy",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda m: (m.get("content") or "").rstrip().endswith("TERMINATE"),
    )
    
    # Register tool for execution
    user_proxy.register_for_execution(name="search_papers")(search_papers)
    
    return user_proxy


# ============================================================================
# STEP 2: GroupChat Setup
# ============================================================================

def make_groupchat() -> GroupChatManager:
    """Create GroupChat with research agent, critic, and user proxy"""
    research_agent = create_research_agent_for_groupchat()
    internal_critic = create_internal_critic()
    user_proxy = create_user_proxy_for_groupchat()
    
    group = GroupChat(
        agents=[user_proxy, research_agent, internal_critic],
        messages=[],
        max_round=15,  # Allow more rounds for tool calls + critique
        speaker_selection_method="round_robin",
    )
    
    manager = GroupChatManager(
        groupchat=group,
        llm_config=LLM_CONFIG,
    )
    
    return manager


def run_with_internal_critic(user_request: str) -> Dict:
    """
    Run the user request through the GroupChat with internal critic.
    
    Returns:
        Dictionary with final_answer and full conversation trace
    """
    try:
        manager = make_groupchat()
        user_proxy = manager.groupchat.agents[0]  # Get the user_proxy from groupchat
        
        init_message = (
            "USER_REQUEST:\n"
            f"{user_request}\n\n"
            "Workflow for agents:\n"
            "- research_agent: read USER_REQUEST, use search_papers tool, "
            "then propose answer as 'DRAFT: ...'.\n"
            "- internal_critic: when you see a DRAFT, respond with 'OK:' or 'CRITIQUE:'.\n"
            "- research_agent: if you get CRITIQUE, revise and send new 'DRAFT:'.\n"
            "- When internal_critic is satisfied, research_agent sends "
            "'FINAL_ANSWER: ...' and includes 'TERMINATE' in the same message.\n\n"
            "The human will only see the FINAL_ANSWER.\n"
        )
        
        final = user_proxy.initiate_chat(
            manager,
            message=init_message,
        )
        
        trace = list(manager.groupchat.messages)
        
        # Extract the FINAL_ANSWER from research_agent
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
    except Exception as e:
        import traceback
        print(f"\n!!! FULL ERROR TRACEBACK !!!")
        traceback.print_exc()
        raise


# ============================================================================
# STEP 3: External LLM-as-Judge
# ============================================================================

def create_judge_agent() -> ConversableAgent:
    """External judge that evaluates final answers"""
    return ConversableAgent(
        name="judge_agent",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are an external evaluator of a research paper search agent.\n"
            "Return STRICT JSON only, no extra commentary.\n\n"
            "Fields to include:\n"
            "- completeness (1-5): Did the agent address all parts of the request? "
            "(topic, year, citations)\n"
            "- accuracy (1-5): Are the search parameters and filters correct?\n"
            "- quality (1-5): Are results presented clearly with relevant details?\n"
            "- robustness (1-5): Does it handle ambiguous or edge case requests well?\n"
            "- feedback (string): Brief explanation of the scores\n\n"
            "Scoring guide:\n"
            "5 = Excellent, 4 = Good, 3 = Acceptable, 2 = Poor, 1 = Failed\n"
        ),
    )


def build_judge_prompt(user_prompt: str, final_answer: str) -> str:
    """Build the prompt for the judge agent"""
    return f"""
You are evaluating a research paper search agent's performance.

User prompt:
\"\"\"{user_prompt}\"\"\"

Final answer from the agent (after internal critic review):
\"\"\"{final_answer}\"\"\"

Evaluate according to your criteria and return JSON ONLY in this exact format:
{{
    "completeness": <1-5>,
    "accuracy": <1-5>,
    "quality": <1-5>,
    "robustness": <1-5>,
    "feedback": "<brief explanation>"
}}
"""


def llm_judge_score(user_prompt: str, final_answer: str) -> Dict:
    """
    Use LLM-as-judge to score the agent's final answer.
    
    Returns:
        Dictionary with scores and feedback
    """
    judge = create_judge_agent()
    judge_prompt = build_judge_prompt(user_prompt, final_answer)
    
    # Get judge's response
    response = judge.generate_reply(
        messages=[{"role": "user", "content": judge_prompt}]
    )
    
    # Extract JSON from response
    content = response if isinstance(response, str) else response.get("content", "")
    
    # Try to parse JSON
    try:
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        return json.loads(content.strip())
    except json.JSONDecodeError as e:
        print(f"Failed to parse judge response: {e}")
        print(f"Raw content: {content}")
        return {
            "completeness": 0,
            "accuracy": 0,
            "quality": 0,
            "robustness": 0,
            "feedback": f"Error parsing judge response: {e}"
        }


# ============================================================================
# STEP 4: Test Prompts
# ============================================================================

TEST_PROMPTS = [
    "Find recent research on neural networks",
    
    # Complex requests
    "Find papers on transformer architecture from 2017 onwards that have significantly impacted NLP",
    
    # Edge cases
    "Search for papers about quantum computing from 2030",
    "Find papers with exactly 0 citations from last year",
 
]


# ============================================================================
# STEP 5: Batch Evaluation
# ============================================================================

def evaluate_prompt(prompt: str) -> Dict:
    """
    Evaluate a single prompt through the full pipeline:
    1. Run through GroupChat with internal critic
    2. Score with external LLM judge
    
    Returns:
        Dictionary with prompt, final answer, and scores
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {prompt}")
    print('='*80)
    
    try:
        # 1) Run through GroupChat with internal critic
        internal_result = run_with_internal_critic(prompt)
        final_answer = internal_result["final_answer"]
        
        print(f"\nFinal answer received (length: {len(final_answer)} chars)")
        
        # 2) External judge scores the final answer
        judge_scores = llm_judge_score(prompt, final_answer)
        
        return {
            "prompt": prompt,
            "final_answer": final_answer,
            "judge_scores": judge_scores,
            "conversation_trace": internal_result["trace"]
        }
    except Exception as e:
        import traceback
        print(f"\n!!! FULL ERROR TRACEBACK FOR PROMPT: {prompt} !!!")
        traceback.print_exc()
        print(f"!!! END TRACEBACK !!!\n")
        return {
            "prompt": prompt,
            "final_answer": f"ERROR: {e}",
            "judge_scores": {
                "completeness": 0,
                "accuracy": 0,
                "quality": 0,
                "robustness": 0,
                "feedback": f"Evaluation failed: {e}"
            }
        }


def run_batch_evaluation(prompts: List[str] = None) -> List[Dict]:
    """
    Run batch evaluation on all test prompts.
    
    Returns:
        List of evaluation results
    """
    if prompts is None:
        prompts = TEST_PROMPTS
    
    results = []
    for prompt in prompts:
        try:
            result = evaluate_prompt(prompt)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating prompt '{prompt}': {e}")
            results.append({
                "prompt": prompt,
                "final_answer": f"ERROR: {e}",
                "judge_scores": {
                    "completeness": 0,
                    "accuracy": 0,
                    "quality": 0,
                    "robustness": 0,
                    "feedback": f"Evaluation failed: {e}"
                }
            })
    
    return results


def print_summary(results: List[Dict]):
    """Print summary statistics of evaluation results"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Per-prompt results
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Prompt: {r['prompt']}")
        print(f"   Scores: {r['judge_scores']}")
    
    # Aggregate statistics
    def avg(key: str) -> float:
        valid_scores = [r["judge_scores"][key] for r in results 
                       if isinstance(r["judge_scores"].get(key), (int, float))]
        return mean(valid_scores) if valid_scores else 0.0
    
    print("\n" + "="*80)
    print("AGGREGATE SCORES (out of 5)")
    print("="*80)
    print(f"Average Completeness: {avg('completeness'):.2f}")
    print(f"Average Accuracy:     {avg('accuracy'):.2f}")
    print(f"Average Quality:      {avg('quality'):.2f}")
    print(f"Average Robustness:   {avg('robustness'):.2f}")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Starting Research Agent Evaluation")
    print("This will test both internal (critic) and external (judge) evaluation\n")
    
    # Run batch evaluation
    results = run_batch_evaluation()
    
    # Print summary
    print_summary(results)
    
    # Optionally save results to file
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to evaluation_results.json")