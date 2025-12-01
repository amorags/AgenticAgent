from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
# NOTE: Assuming research_agent.agent.config.LLM_CONFIG is available
from research_agent.agent.config import LLM_CONFIG
# NOTE: Assuming research_agent.tools.search_paper.search_papers is available
from research_agent.tools.search_paper import search_papers
from typing import List, Dict
import json
from statistics import mean

# ============================================================================
# STEP 1: Create Agents
# ============================================================================

paper_agent = AssistantAgent(
    name="paper_agent",
    llm_config={
        **LLM_CONFIG,
        "functions": [
            {
                "name": "search_papers",
                "description": "Search for academic papers from Semantic Scholar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Research topic to search for"
                        },
                        "min_citations": {
                            "type": "integer",
                            "description": "Minimum citation count"
                        },
                        "year": {
                            "type": "integer",
                            "description": "Publication year (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 10)"
                        }
                    },
                    "required": ["topic"]
                }
            }
        ]
    },
    system_message=(
        "You are a research paper search assistant.\n\n"
        "**STRICT REQUIREMENTS:**\n"
        "1. You MUST call search_papers function BEFORE writing any draft\n"
        "2. Never make up papers - only use real search results\n"
        "3. After receiving search results, write a CONCISE draft with MAX 5 papers\n"
        "4. For authors: show ONLY first 3 authors, then add 'et al.' if more exist\n"
        "5. Format each paper clearly with: Title, First 3 Authors (et al.), Year, Citations, URL\n\n"
        "**Process:**\n"
        "1. Call search_papers with user's criteria\n"
        "2. Wait for real results from tool execution\n"
        "3. Write 'DRAFT:' or 'NEW DRAFT:' with TOP 5 formatted results (even if you got more)\n"
        "4. Wait for and respond to critic feedback\n"
        "5. When critic says 'OK:', write 'FINAL_ANSWER:' with the approved content and 'TERMINATE'\n\n"
        "**Example Format:**\n"
        "DRAFT:\n"
        "Here are 5 highly-cited papers on [topic]:\n\n"
        "1. **Title Here**\n"
        "   - Authors: First Author, Second Author, Third Author et al.\n"
        "   - Year: 2020 | Citations: 5000\n"
        "   - URL: https://...\n"
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

# internal_critic: Defined as UserProxyAgent for robust flow control
# Removed the duplicate definition that was further down.
internal_critic = AssistantAgent(
    name="internal_critic",
    llm_config=LLM_CONFIG,
    system_message=(
        "You are an internal critic reviewing research_agent's DRAFTs.\n\n"
        "**YOUR ROLE:**\n"
        "- ONLY respond when you see a message containing 'DRAFT:'\n"
        "- IGNORE tool calls and tool results\n"
        "- Provide immediate feedback on drafts\n\n"
        "**Evaluation Criteria:**\n"
        "1. Are there exactly 5 or fewer papers listed?\n"
        "2. Do all papers meet the citation requirement (if specified)?\n"
        "3. Are author lists concise (max 3 authors shown + et al.)?\n"
        "4. Are URLs provided for each paper?\n"
        "5. Is the format clear and consistent?\n\n"
        "**Response Format:**\n"
        "- If acceptable: 'OK: [brief reason]'\n"
        "- If issues found: 'CRITIQUE: [specific issue]'\n\n"
        "Be concise. Approve good drafts quickly.\n"
    ),
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    llm_config=False,
    human_input_mode="NEVER",
    # Termination is strictly when 'FINAL_ANSWER:' + 'TERMINATE' is received.
    is_termination_msg=lambda m: "TERMINATE" in m.get("content", ""),
    code_execution_config=False,
)
user_proxy.register_for_execution(name="search_papers")(search_papers)

# ============================================================================
# STEP 2: Custom Speaker Selection
# ============================================================================

def custom_speaker_selection(last_speaker, groupchat):
    """
    Defensive logic to enforce the Tool Call -> Draft -> Critique flow, 
    using the user_proxy as a 'sanitizer' only when a DRAFT is detected.
    """
    messages = groupchat.messages
    
    if not messages or last_speaker is None:
        # 0. Initial check: Start the conversation
        return paper_agent

    last_msg = messages[-1]
    last_content = last_msg.get("content", "")

    # ===================================================================
    # 1. TOOL CALL FLOW: paper_agent <-> user_proxy
    # ===================================================================

    # A. paper_agent calls tool -> user_proxy executes
    if last_speaker is paper_agent and last_msg.get("tool_calls"):
        return user_proxy
    
    # B. user_proxy returns tool result -> paper_agent processes result
    if last_speaker is user_proxy and last_msg.get("role") == "tool":
        return paper_agent
    
    # ===================================================================
    # 2. CRITIQUE FLOW: paper_agent -> user_proxy -> internal_critic -> paper_agent
    # ===================================================================
    
    # C. paper_agent sends DRAFT/Revision -> user_proxy (Sanitizer)
    if last_speaker is paper_agent and ("DRAFT:" in last_content or "NEW DRAFT:" in last_content):
        return user_proxy
        
    # D. user_proxy (Sanitizer) -> internal_critic (Reviewer)
    # If user_proxy spoke, and it wasn't a tool result (handled by B), 
    # it must be the sanitizer turn, so next is the critic.
    if last_speaker is user_proxy and last_msg.get("role") != "tool":
        return internal_critic

    # E. internal_critic (Critique) -> paper_agent (Reviser)
    if last_speaker is internal_critic:
        return paper_agent

    # ===================================================================
    # 3. FALLBACK
    # ===================================================================

    # Default to paper_agent if the flow is broken or starting
    return paper_agent
    
# ============================================================================
# STEP 3: Create GroupChat
# ============================================================================

def make_groupchat() -> GroupChatManager:
    group = GroupChat(
        agents=[user_proxy, paper_agent, internal_critic],
        messages=[],
        max_round=25,
        speaker_selection_method=custom_speaker_selection,
    )
    manager = GroupChatManager(groupchat=group, llm_config=LLM_CONFIG)
    return manager

# ============================================================================
# STEP 4: Run User Request Through GroupChat
# ============================================================================

def run_with_internal_critic(user_request: str) -> Dict:
    manager = make_groupchat()

    init_message = (
        f"USER_REQUEST:\n{user_request}\n\n"
        "Workflow:\n"
        "1. paper_agent: Call search_papers, then write 'DRAFT:' with TOP 5 results\n"
        "2. internal_critic: Review the DRAFT and respond 'OK:' or 'CRITIQUE:'\n"
        "3. paper_agent: If CRITIQUE, revise and send 'NEW DRAFT:'\n"
        "4. When internal_critic says 'OK:', paper_agent sends 'FINAL_ANSWER:' + 'TERMINATE'\n"
    )

    try:
        final = user_proxy.initiate_chat(manager, message=init_message)
        trace = list(manager.groupchat.messages)
    except Exception as e:
        print(f"Chat error: {e}")
        trace = list(manager.groupchat.messages) if hasattr(manager, 'groupchat') else []
        final = None

    # Extract FINAL_ANSWER (MUST contain "TERMINATE")
    final_answer = None
    for msg in reversed(trace):
        if msg.get("name") == "paper_agent":
            content = msg.get("content", "")
            # Check for both FINAL_ANSWER: and TERMINATE to ensure successful completion
            if isinstance(content, str) and "FINAL_ANSWER:" in content and "TERMINATE" in content:
                final_answer = content
                break
    
    # NEW Fallback: If chat did not complete successfully, find the last draft 
    # and wrap it in an error message for diagnostic scoring.
    if not final_answer:
        last_draft = None
        for msg in reversed(trace):
            if msg.get("name") == "paper_agent":
                content = msg.get("content", "")
                if isinstance(content, str) and ("DRAFT:" in content or "NEW DRAFT:" in content):
                    last_draft = content
                    break
        
        if last_draft:
            # Signal to the judge that the run failed at the end of a draft cycle.
            final_answer = (
                "ERROR: Chat terminated prematurely before critic approval. "
                "Scoring the last incomplete DRAFT:\n\n"
            ) + last_draft
        else:
            final_answer = "ERROR: No response or final answer generated."


    return {
        "final_answer": final_answer,
        "trace": trace,
    }

# ============================================================================
# STEP 5: External Judge
# ============================================================================

def create_judge_agent() -> AssistantAgent:
    return AssistantAgent(
        name="judge_agent",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are a strict evaluation agent. You will be given:\n"
            "1) PROMPT – the user's original request\n"
            "2) ANSWER – the research agent's final response\n\n"
            "Evaluate the ANSWER and return ONLY valid JSON:\n"
            "{\n"
            '  "completeness": 1-5,  // How fully ANSWER addresses PROMPT\n'
            '  "accuracy": 1-5,      // Factual correctness, proper citations\n'
            '  "quality": 1-5,       // Clarity, structure, readability\n'
            '  "robustness": 1-5,    // Proper tool usage, edge case handling\n'
            '  "feedback": "string"  // Concise explanation\n'
            "}\n\n"
            "Requirements:\n"
            "- Output ONLY valid JSON, nothing else\n"
            "- Use integers 1-5, never decimals\n"
            "- Be strict: missing info, inaccuracies, or poor format = lower score\n"
            "- Check: Are papers real? Citations accurate? Format clean? Author lists concise?\n"
        ),
    )

def llm_judge_score(user_prompt: str, final_answer: str) -> Dict:
    """Evaluate using a fresh agent"""
    judge = create_judge_agent()
    judge_proxy = UserProxyAgent(
        name="judge_proxy",
        llm_config=False,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0
    )
    
    eval_prompt = (
        f'PROMPT: "{user_prompt}"\n\n'
        f'ANSWER: "{final_answer}"\n\n'
        f'Return JSON only: completeness, accuracy, quality, robustness, feedback'
    )
    
    try:
        # Note: max_turns=1 is correct for a single-turn evaluation
        judge_proxy.initiate_chat(judge, message=eval_prompt, max_turns=1)
        last_msg = judge.last_message()
        content = last_msg.get("content", "{}") if last_msg else "{}"
        
        # Clean potential markdown
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        return {
            "completeness": 0,
            "accuracy": 0,
            "quality": 0,
            "robustness": 0,
            "feedback": f"JSON parse error: {e}. Raw content: {content}"
        }
    except Exception as e:
        return {
            "completeness": 0,
            "accuracy": 0,
            "quality": 0,
            "robustness": 0,
            "feedback": f"API error occurred: {str(e)}"
        }

# ============================================================================
# STEP 6: Evaluation
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

def print_summary(results: List[Dict]):
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['prompt']}")
        print(f"   Scores: {r['judge_scores']}")

    def avg(key: str) -> float:
        valid = [r["judge_scores"][key] for r in results 
                 if isinstance(r["judge_scores"].get(key), (int, float))]
        return mean(valid) if valid else 0.0

    print("\n" + "="*80)
    print(f"Avg Completeness: {avg('completeness'):.2f}/5")
    print(f"Avg Accuracy:     {avg('accuracy'):.2f}/5")
    print(f"Avg Quality:      {avg('quality'):.2f}/5")
    print(f"Avg Robustness:   {avg('robustness'):.2f}/5")
    print("="*80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Research Agent Evaluation WITH GroupChat + Internal Critic\n")
    
    # Get user input
    print("Enter your research request (or press Enter for default):")
    user_input = input("> ").strip()
    
    if not user_input:
        user_input = "Find research on reinforcement learning with 1000+ citations"
        print(f"Using default: {user_input}\n")
    
    # Run single evaluation
    try:
        result = evaluate_prompt(user_input)
        print_summary([result])
        
        with open("tool_evaluation_results.json", "w") as f:
            json.dump([result], f, indent=2)
        
        print("\n✅ Results saved to tool_evaluation_results.json")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()