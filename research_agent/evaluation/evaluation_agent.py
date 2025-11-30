"""
Agent Evaluation WITH Tool Calling
Properly implements search_papers tool usage
"""

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from research_agent.agent.config import LLM_CONFIG
from research_agent.tools.search_paper import search_papers
from typing import Dict, List
import json
from statistics import mean


# ============================================================================
# STEP 1: Define Agents WITH TOOL CALLING
# ============================================================================

def create_research_agent() -> AssistantAgent:
    """Research agent that MUST use search_papers tool"""
    agent = AssistantAgent(
        name="research_agent",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are a research paper search assistant.\n\n"
            "IMPORTANT: You MUST use the search_papers function to find papers. "
            "DO NOT make up papers from your knowledge.\n\n"
            "Workflow:\n"
            "1) Read the USER_REQUEST carefully\n"
            "2) Call search_papers() with appropriate parameters:\n"
            "   - topic: the research topic\n"
            "   - year: specific year if mentioned\n"
            "   - year_filter: 'exact', 'before', or 'after'\n"
            "   - min_citations: minimum citations if mentioned\n"
            "   - limit: number of results (default 5)\n"
            "3) After getting results, format as 'DRAFT:' with paper details\n"
            "4) If critic says CRITIQUE, revise. If OK, send 'FINAL_ANSWER:' with TERMINATE\n\n"
            "Example: For 'papers about ML from 2020 with 100+ citations':\n"
            "Call: search_papers(topic='machine learning', year=2020, year_filter='exact', min_citations=100)\n"
        ),
    )
    
    # Register tool for LLM to know about it
    agent.register_for_llm(
        name="search_papers",
        description=(
            "Search for academic research papers. Returns real papers from Semantic Scholar.\n"
            "Parameters:\n"
            "- topic (str, required): Research topic to search\n"
            "- year (int, optional): Year to filter by\n"
            "- year_filter (str, optional): 'exact', 'before', or 'after' (default 'exact')\n"
            "- min_citations (int, optional): Minimum citation count\n"
            "- limit (int, optional): Max results (default 5)"
        )
    )(search_papers)
    
    return agent


def create_internal_critic() -> AssistantAgent:
    """Internal critic - does NOT use tools, only reviews"""
    return AssistantAgent(
        name="internal_critic",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are an internal critic. Review research_agent's DRAFT answers.\n\n"
            "Check:\n"
            "- Did agent use search_papers tool?\n"
            "- Are results complete (topic, year, citations addressed)?\n"
            "- Are parameters correct?\n"
            "- Is presentation clear?\n\n"
            "Respond:\n"
            "- 'OK: <reason>' if acceptable\n"
            "- 'CRITIQUE: <issue> + <fix>' if problems\n\n"
            "After 2 critiques, be lenient to avoid loops.\n"
        ),
    )


def create_user_proxy() -> UserProxyAgent:
    """User proxy - executes the search_papers tool"""
    user_proxy = UserProxyAgent(
        name="user_proxy",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", ""),
        max_consecutive_auto_reply=10,
    )
    
    # Register tool for EXECUTION
    user_proxy.register_for_execution(name="search_papers")(search_papers)
    
    return user_proxy


# ============================================================================
# STEP 2: Two-Agent Setup (NO GroupChat to avoid Mistral issues)
# ============================================================================

def run_with_tool_calling(user_request: str) -> Dict:
    """
    Run request with tool calling using simple 2-agent setup.
    Avoids GroupChat complexity that causes Mistral API errors.
    """
    try:
        research_agent = create_research_agent()
        user_proxy = create_user_proxy()
        
        # Simple 2-agent conversation
        chat_result = user_proxy.initiate_chat(
            research_agent,
            message=f"USER_REQUEST: {user_request}\n\nUse search_papers to find relevant papers and provide a detailed answer.",
            max_turns=3,
        )
        
        # Get final message
        messages = chat_result.chat_history if hasattr(chat_result, 'chat_history') else []
        final_answer = messages[-1].get("content", "No response") if messages else "No response"
        
        return {
            "final_answer": final_answer,
            "trace": messages,
            "tool_called": any("search_papers" in str(m) for m in messages)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "final_answer": f"ERROR: {e}",
            "trace": [],
            "tool_called": False
        }


def run_with_internal_critic(user_request: str) -> Dict:
    """
    Alternative: Three-agent with manual orchestration (no GroupChat).
    More control, avoids Mistral message ordering issues.
    """
    try:
        research_agent = create_research_agent()
        critic = create_internal_critic()
        user_proxy = create_user_proxy()
        
        conversation = []
        
        # Round 1: User -> Research Agent
        print(f"\n🔹 Round 1: Asking research agent...")
        response1 = user_proxy.initiate_chat(
            research_agent,
            message=f"USER_REQUEST: {user_request}\n\nUse search_papers tool and format response as 'DRAFT: ...'",
            max_turns=2,
            clear_history=True,
        )
        draft1 = response1.chat_history[-1].get("content", "") if response1.chat_history else ""
        conversation.append({"round": 1, "from": "research_agent", "content": draft1})
        
        # Round 2: Critic reviews
        print(f"🔹 Round 2: Critic reviewing...")
        critique_response = critic.generate_reply(
            messages=[{"role": "user", "content": f"Review this DRAFT:\n{draft1}\n\nRespond with 'OK:' or 'CRITIQUE:'"}]
        )
        critique = critique_response if isinstance(critique_response, str) else critique_response.get("content", "")
        conversation.append({"round": 2, "from": "critic", "content": critique})
        
        final_answer = draft1
        
        # Round 3: If critique, ask agent to revise
        if "CRITIQUE:" in critique:
            print(f"🔹 Round 3: Agent revising based on critique...")
            response2 = user_proxy.initiate_chat(
                research_agent,
                message=f"The critic said: {critique}\n\nPlease revise your answer and format as 'FINAL_ANSWER: ...' with TERMINATE",
                max_turns=2,
                clear_history=True,
            )
            final_answer = response2.chat_history[-1].get("content", "") if response2.chat_history else draft1
            conversation.append({"round": 3, "from": "research_agent", "content": final_answer})
        
        return {
            "final_answer": final_answer,
            "trace": conversation,
            "tool_called": "search_papers" in str(conversation)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "final_answer": f"ERROR: {e}",
            "trace": [],
            "tool_called": False
        }


# ============================================================================
# STEP 3: External Judge
# ============================================================================

def create_judge_agent() -> AssistantAgent:
    """External judge"""
    return AssistantAgent(
        name="judge_agent",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are an evaluator. Return ONLY valid JSON.\n\n"
            "Evaluate:\n"
            "- completeness (1-5)\n"
            "- accuracy (1-5)\n"
            "- quality (1-5)\n"
            "- robustness (1-5)\n"
            "- feedback (string)\n\n"
            'Return: {"completeness": 4, "accuracy": 5, "quality": 4, '
            '"robustness": 3, "feedback": "..."}'
        ),
    )


def llm_judge_score(user_prompt: str, final_answer: str) -> Dict:
    """External judge scoring"""
    judge = create_judge_agent()
    
    try:
        response = judge.generate_reply(
            messages=[{"role": "user", "content": f'Request: "{user_prompt}"\nAnswer: "{final_answer}"\n\nReturn JSON only.'}]
        )
        content = response if isinstance(response, str) else response.get("content", "")
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        return json.loads(content.strip())
    except Exception as e:
        return {
            "completeness": 0, "accuracy": 0, "quality": 0, "robustness": 0,
            "feedback": f"Judge error: {e}"
        }


# ============================================================================
# STEP 4: Test Prompts
# ============================================================================

TEST_PROMPTS = [
    "Find papers about machine learning from 2020 with at least 500 citations",
    "Search for deep learning papers published after 2018",
    "I need papers about neural networks",
]


# ============================================================================
# STEP 5: Evaluation
# ============================================================================

def evaluate_prompt(prompt: str, use_critic: bool = True) -> Dict:
    """Evaluate a single prompt"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {prompt}")
    print('='*80)
    
    # Run with or without critic
    if use_critic:
        result = run_with_internal_critic(prompt)
    else:
        result = run_with_tool_calling(prompt)
    
    final_answer = result["final_answer"]
    print(f"\n✅ Tool called: {result['tool_called']}")
    print(f"📝 Answer length: {len(final_answer)} chars")
    
    # External judge
    judge_scores = llm_judge_score(prompt, final_answer)
    
    return {
        "prompt": prompt,
        "final_answer": final_answer,
        "judge_scores": judge_scores,
        "tool_called": result["tool_called"],
        "rounds": len(result["trace"])
    }


def run_batch_evaluation(prompts: List[str] = None, use_critic: bool = True) -> List[Dict]:
    """Run batch evaluation"""
    if prompts is None:
        prompts = TEST_PROMPTS
    
    results = []
    for prompt in prompts:
        try:
            result = evaluate_prompt(prompt, use_critic)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "prompt": prompt,
                "final_answer": f"ERROR: {e}",
                "judge_scores": {"completeness": 0, "accuracy": 0, "quality": 0, "robustness": 0, "feedback": str(e)},
                "tool_called": False,
                "rounds": 0
            })
    
    return results


def print_summary(results: List[Dict]):
    """Print summary"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['prompt']}")
        print(f"   Tool Called: {r.get('tool_called', False)}")
        print(f"   Scores: {r['judge_scores']}")
    
    def avg(key: str) -> float:
        valid = [r["judge_scores"][key] for r in results if isinstance(r["judge_scores"].get(key), (int, float))]
        return mean(valid) if valid else 0.0
    
    print("\n" + "="*80)
    print(f"Completeness: {avg('completeness'):.2f}/5")
    print(f"Accuracy:     {avg('accuracy'):.2f}/5")
    print(f"Quality:      {avg('quality'):.2f}/5")
    print(f"Robustness:   {avg('robustness'):.2f}/5")
    print(f"Tools Called: {sum(r.get('tool_called', False) for r in results)}/{len(results)}")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Research Agent Evaluation WITH Tool Calling")
    print("Using manual orchestration to avoid GroupChat/Mistral issues\n")
    
    # Run with internal critic
    results = run_batch_evaluation(use_critic=True)
    print_summary(results)
    
    with open("tool_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to tool_evaluation_results.json")