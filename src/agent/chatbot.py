from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class Chatbot:
    """
    A simple baseline chatbot without agent reasoning.
    This is used for comparison against the ReAct Agent to demonstrate
    the limitations of direct LLM responses for multi-step reasoning tasks.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.history = []

    def get_system_prompt(self) -> str:
        """
        System prompt for the simple chatbot.
        This is a general-purpose travel assistant without tool instructions.
        """
        system_prompt = """
        Bạn là một trợ lý về du lịch, hỗ trợ trong việc gợi ý các địa điểm du lịch dựa trên khu vực mà người dùng muốn đi.

        Dựa trên kiến thức của bạn, hãy đưa ra gợi ý về các địa điểm du lịch phù hợp.

        Hãy tuân theo chỉ dẫn sau:
        - Hãy đưa ra ít nhất 3 gợi ý về địa điểm du lịch.
        - Với mỗi địa điểm du lịch mà bạn gợi ý, hãy cung cấp thông tin chi tiết bao gồm:
            + Tên địa điểm
            + Mô tả ngắn gọn về địa điểm
            + Lý do tại sao địa điểm này phù hợp với yêu cầu của người dùng
            + Thông tin về khách sạn gần đó (nếu có thể)
            + Tips để có thể tận hưởng chuyến đi tốt nhất (nếu có thể)

        Lưu ý: Bạn không có quyền truy cập vào công cụ nào, chỉ có thể sử dụng kiến thức của mình.
        """
        return system_prompt

    def run(self, user_input: str) -> str:
        """
        Run the chatbot: take user input and return LLM response.
        No tool calling, no reasoning loop - just direct response.
        """
        self.history = []
        logger.log_event("CHATBOT_START", {"input": user_input, "model": self.llm.model_name})

        # Append user input to history
        self.history.append({"role": "user", "content": user_input})

        # Generate response directly from LLM (no agent loop)
        result = self.llm.generate_response(
            system_prompt=self.get_system_prompt(),
            history=self.history,
            prompt=user_input
        )
        response = result["content"]

        # Track metrics
        usage = result.get("usage", {})
        latency = result.get("latency_ms", 0)
        provider = result.get("provider", "unknown")
        tracker.track_request(provider, self.llm.model_name, usage, latency)

        # Log the response
        logger.log_event("CHATBOT_RESPONSE", {"response": response, "tokens": usage.get("total_tokens", 0), "latency_ms": latency})
        self.history.append({"role": "assistant", "content": response})

        logger.log_event("CHATBOT_END", {"total_messages": len(self.history)})
        return response

    def chat(self) -> None:
        """
        Interactive chat mode: loop until user exits.
        """
        print("=" * 60)
        print("🤖 Travel Chatbot (Baseline - No Tools)")
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
                
                response = self.run(user_input)
                print("\nAssistant:", response, "\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Chatbot error: {e}")
