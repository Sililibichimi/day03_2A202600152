"""
Demo script: Compare Chatbot vs ReAct Agent

This script demonstrates the difference between:
1. Simple Chatbot: Direct LLM responses (limited reasoning)
2. ReAct Agent: Multi-step reasoning with tool calls
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.openai_provider import OpenAIProvider
from src.core.gemini_provider import GeminiProvider
from src.agent.chatbot import Chatbot
from src.agent.agent import ReActAgent
from src.tools import get_all_tools


def get_llm_provider(provider_name: str = "openai"):
    """Get the configured LLM provider."""
    load_dotenv()
    
    provider_name = provider_name.lower()
    
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        return OpenAIProvider(api_key=api_key)
    
    elif provider_name == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        return GeminiProvider(api_key=api_key)
    
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Use 'openai' or 'gemini'.")


def compare_chatbot_vs_agent():
    """Compare chatbot and agent on the same queries."""
    load_dotenv()
    provider_name = os.getenv("DEFAULT_PROVIDER", "openai")
    
    print("=" * 70)
    print("🤖 CHATBOT VS REACT AGENT COMPARISON")
    print("=" * 70)
    print(f"Provider: {provider_name.upper()}\n")
    
    try:
        llm = get_llm_provider(provider_name)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set up your API keys in .env file.")
        return
    
    # Initialize chatbot and agent
    chatbot = Chatbot(llm=llm)
    agent = ReActAgent(llm=llm, tools=get_all_tools(), max_steps=5)
    
    # Test queries that benefit from multi-step reasoning
    test_queries = [
        "Tôi muốn đi du lịch ở Hà Nội. Hôm nay thời tiết như thế nào? Có phòng khách sạn nào trống không?",
        "Giúp tôi tìm một điểm du lịch ở Da Nang với thời tiết đẹp và giá phòng rẻ.",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST CASE {i}: {query[:60]}...")
        print('=' * 70)
        
        # Chatbot response
        print("\n🤖 CHATBOT (No Tools):")
        print("-" * 70)
        try:
            chatbot_response = chatbot.run(query)
            print(chatbot_response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Agent response
        print("\n🦾 REACT AGENT (With Tools):")
        print("-" * 70)
        try:
            agent_response = agent.run(query)
            print(agent_response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()


def run_chatbot_interactive():
    """Run the chatbot in interactive mode."""
    load_dotenv()
    provider_name = os.getenv("DEFAULT_PROVIDER", "openai")
    
    try:
        llm = get_llm_provider(provider_name)
        chatbot = Chatbot(llm=llm)
        chatbot.chat()
    except ValueError as e:
        print(f"❌ Error: {e}")


def run_agent_interactive():
    """Run the agent in interactive mode."""
    load_dotenv()
    provider_name = os.getenv("DEFAULT_PROVIDER", "openai")
    
    try:
        llm = get_llm_provider(provider_name)
        agent = ReActAgent(llm=llm, tools=get_all_tools(), max_steps=5)
        
        print("=" * 60)
        print("🦾 ReAct Agent (With Tools)")
        print("=" * 60)
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = agent.run(user_input)
                print("\nAgent:", response, "\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    except ValueError as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chatbot and Agent Demo")
    parser.add_argument(
        "--mode",
        choices=["compare", "chatbot", "agent"],
        default="compare",
        help="Mode to run: compare (default), chatbot, or agent"
    )
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        compare_chatbot_vs_agent()
    elif args.mode == "chatbot":
        run_chatbot_interactive()
    elif args.mode == "agent":
        run_agent_interactive()
