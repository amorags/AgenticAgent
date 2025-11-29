from autogen import ConversableAgent
from research_agent.agent.config import LLM_CONFIG


def create_test_agent() -> ConversableAgent:
    """Create a simple test agent to verify API connection"""
    agent = ConversableAgent(
        name="Test Agent",
        system_message="You are a helpful AI assistant. Keep responses brief.",
        llm_config=LLM_CONFIG,
    )
    return agent


def create_user_proxy():
    """Create a user proxy for interaction"""
    user_proxy = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )
    return user_proxy


def main():
    """Test that the API connection works"""
    print("Testing Mistral API connection...")
    
    user_proxy = create_user_proxy()
    test_agent = create_test_agent()
    
    chat_result = user_proxy.initiate_chat(
        test_agent, 
        message="Hello! Please respond with 'API connection successful' and then TERMINATE.",
        max_turns=1
    )
    
    print("\n✅ Test completed successfully!")
    print(chat_result)


if __name__ == "__main__":
    main()