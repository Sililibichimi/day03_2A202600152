import os
import re
from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger

class ReActAgent:
    """
    SKELETON: A ReAct-style Agent that follows the Thought-Action-Observation loop.
    Students should implement the core loop logic and tool execution.
    """
    
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []

    def get_system_prompt(self) -> str:
        
        system_prompt = f"""
        Bạn là một trợ lý về du lịch, hỗ trợ trong việc gợi ý các địa điểm du lịch dựa trên khu vực mà người dùng muốn đi. 
        Bạn có quyền truy cập vào các công cụ sau :
        {"google_search": "Tìm kiếm thông tin trên Google về các địa điểm du lịch ở khu vực mà người dùng yêu cầu."}
        {"weather_api": "Kiểm tra thời tiết hiện tại tại khu vực mà người dùng muốn đến."}
        {"booking_api": "Kiểm tra tình trạng phòng khách sạn tại khu vực mà người dùng muốn đến."}

        Dựa trên thông tin mà người dùng cung cấp, hãy sử dụng các công cụ trên để thu thập thông tin cần thiết và đưa ra gợi ý về các địa điểm du lịch phù hợp.

        Hãy tuân theo chỉ dẫn sau:
        - Hãy đưa ra ít nhất 3 gợi ý về địa điểm du lịch.
        - Với mỗi địa điểm du lịch mà bạn gợi ý, hãy cung cấp thông tin chi tiết bao gồm:
            + Tên địa điểm
            + Mô tả ngắn gọn về địa điểm
            + Lý do tại sao địa điểm này phù hợp với yêu cầu của người dùng(Giá cả, thời tiết, hoạt động giải trí, v.v.)
            + Thông tin về tình trạng phòng khách sạn gần địa điểm đó (nếu có thể) 
            + Tips để có thể tận hưởng chuyến đi tốt nhất (nếu có thể)

        Kết quả trả về tuân theo định dạng sau:
        Thought: suy nghĩ của bạn về yêu cầu của người dùng và cách bạn sẽ sử dụng các công cụ.
        Action: công cụ bạn sẽ sử dụng và các tham số của nó.
        Observation: kết quả trả về từ công cụ sau khi bạn thực hiện Action.
        ... (lặp lại Thought/Action/Observation nếu cần thiết)
        Final response: gợi ý cuối cùng của bạn cho người dùng dựa trên các thông tin thu thập được.
        """

        return system_prompt

    def run(self, user_input: str) -> str:
        """
        TODO: Implement the ReAct loop logic.
        1. Generate Thought + Action.
        2. Parse Action and execute Tool.
        3. Append Observation to prompt and repeat until Final Answer.
        """

        self.history = []

        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        
        current_prompt = user_input
        steps = 0

        while steps < self.max_steps:
            self.history.append(current_prompt)
            logger.log_event("AGENT_STEP", {"step": steps, "prompt": current_prompt})
            response = self.llm.generate(current_prompt, system_prompt=self.get_system_prompt())
            
            
            # Parse Thought/Action from result
            

            # TODO: If Action found -> Call tool -> Append Observation
            
            # TODO: If Final Answer found -> Break loop
            
            steps += 1
            
        logger.log_event("AGENT_END", {"steps": steps})
        return "Not implemented. Fill in the TODOs!"

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Helper method to execute tools by name.
        """
        for tool in self.tools:
            if tool['name'] == tool_name:
                # TODO: Implement dynamic function calling or simple if/else
                fn = tool.get('function') or tool.get('func')
                if not callable(fn): 
                    return f"Tool {tool_name} is not callable."
                
                parse_args = re.split(r'\s*,\s*', args.strip())
                return f"Result of {tool_name}"
        return f"Tool {tool_name} not found."
