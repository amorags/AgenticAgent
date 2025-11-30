from autogen import ConversableAgent
from typing import Annotated, Literal, Optional
from research_agent.agent.config import LLM_CONFIG
from research_agent.tools.search_paper import search_papers


def create_research_agent() -> ConversableAgent:
    """Create a research agent that can search for academic papers"""
    agent = ConversableAgent(
        name="Research Agent",
        system_message=(
            "You are a helpful research assistant that can search for academic papers. "
            "When a user asks for papers, extract the following information:\n"
            "- topic: the research topic to search for\n"
            "- year: the year to filter by (if mentioned)\n"
            "- year_filter: 'exact', 'before', or 'after' (based on user's request)\n"
            "- min_citations: minimum citation count (if mentioned)\n"
            "- limit: how many results to return (default 10)\n\n"
            "Use the search_papers tool to find papers matching the criteria.\n"
            "Present results in a clear, organized format.\n"
            "Return 'TERMINATE' when the task is complete."
        ),
        llm_config=LLM_CONFIG,
    )

    # Register the search_papers tool
    agent.register_for_llm(
        name="search_papers",
        description=(
            "Search for academic research papers. "
            "Parameters: topic (required), year (optional), min_citations (optional), "
            "year_filter ('exact', 'before', 'after'), limit (default 10)"
        )
    )(search_papers)

    return agent


def create_user_proxy():
    """Create a user proxy for interaction"""
    user_proxy = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )
    
    # Register the tool for execution
    user_proxy.register_for_execution(name="search_papers")(search_papers)
    
    return user_proxy


def main():
    """Main function to run the research agent"""
    print("=" * 80)
    print("Research Paper Search Agent")
    print("=" * 80)
    print("\nStarting agent...")
    print("\nYou can ask questions like:")
    print("- 'Find papers about machine learning from 2020 with at least 500 citations'")
    print("\n" + "=" * 80 + "\n")
    
    # Create agents
    user_proxy = create_user_proxy()
    research_agent = create_research_agent()
    
    # Get user query
    user_query = input("Enter your research query (or press Enter for example): ").strip()
    
    if not user_query:
        user_query = "Find papers about deep learning from 2020 with at least 100 citations"
        print(f"\nUsing example query: {user_query}\n")
    
    # Start the conversation
    chat_result = user_proxy.initiate_chat(
        research_agent,
        message=user_query,
        max_turns=5
    )
    
    print("\n" + "=" * 80)
    print("Search completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()