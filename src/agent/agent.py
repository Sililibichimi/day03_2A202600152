import ast
import re
from typing import List, Dict, Any, Optional, Tuple
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
        Execute the agent loop.
        """
        self.history = []
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        prompt = user_input
        steps = 0

        while steps < self.max_steps:
            # TODO: Generate LLM response
            response = self.llm.generate_response(self.get_system_prompt(), self.history, prompt)
            self.history.append({"role": "assistant", "content": response})
            logger.log_event("AGENT_STEP",{"step":steps,"response": response})
            # TODO: Parse Thought/Action from result
            action_match = re.search(r"Action:\s*(\w+)\[([^\]]*)\]", response)
            final_match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)



            # TODO: If Action found -> Call tool -> Append Observation
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()
                logger.log_event("TOOL_CALL", {"tool": tool_name, "input": tool_input})

            if tool_name in self.tools:
                observation = self.tools[tool_name].run(tool_input)
            else:
                observation = f"Error: tool '{tool_name}' not found."

            logger.log_event("OBSERVATION", {"observation": observation})
            prompt = f"Observation: {observation}"
            self.history.append({"role": "user", "content": prompt})

            steps += 1
            # TODO: If Final Answer found -> Break loop
            if final_match:
                final_answer = final_match.group(1).strip()
                logger.log_event("FINAL_ANSWER", {"answer": final_answer, "steps": steps})
            return final_answer
            
        logger.log_event("AGENT_END", {"steps": steps})
        return "Not implemented. Fill in the TODOs!"

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Find a tool by name and call its function with parsed arguments.
        """
        for tool in self.tools:
            if tool.get("name") == tool_name:
                tool_func = tool.get("func")
                if not callable(tool_func):
                    return f"Tool '{tool_name}' is present but has no callable func."

                parsed_args = self._parse_tool_arguments(args)
                try:
                    if isinstance(parsed_args, tuple):
                        return str(tool_func(*parsed_args))
                    if isinstance(parsed_args, list):
                        return str(tool_func(*parsed_args))
                    if parsed_args == "" or parsed_args is None:
                        return str(tool_func())
                    return str(tool_func(parsed_args))
                except Exception as exc:
                    logger.error(f"Error executing tool {tool_name}: {exc}")
                    return f"Error executing tool {tool_name}: {exc}"

        return f"Tool {tool_name} not found."

    def _parse_tool_arguments(self, args: str) -> Any:
        """
        Convert a string of arguments into Python types when possible.
        """
        if not args:
            return ""

        sanitized = args.strip()
        try:
            parsed = ast.literal_eval(sanitized)
            return parsed
        except (ValueError, SyntaxError):
            return sanitized
