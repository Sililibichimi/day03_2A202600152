import ast
import re
from typing import List, Dict, Any, Optional, Tuple
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger


class ReActAgent:
    def __init__(self, llm, tools, max_steps=5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []

    def get_system_prompt(self):
        return """
Ban la mot tro ly ve du lich, ho tro trong viec goi y cac dia diem du lich.
Cac cong cu:
- google_search: Tim kiem thong tin tren Google
- weather_api: Kiem tra thoi tiet
- booking_api: Kiem tra phong khach san

Dinh dang phan hoi:
Thought: suy nghi cua ban
Action: ten_cong_cu[tham_so]
Observation: ket qua tu cong cu
Final Answer: cau tra loi cuoi cung
"""

    def run(self, user_input):
        self.history = []
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        prompt = user_input
        steps = 0
        while steps < self.max_steps:
            response = self.llm.generate_response(self.get_system_prompt(), self.history, prompt)
            self.history.append({"role": "assistant", "content": response})
            logger.log_event("AGENT_STEP", {"step": steps, "response": response})

            action_match = re.search(r"Action:\s*(\w+)\[([^\]]*)\]", response)
            final_match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)

            if final_match:
                final_answer = final_match.group(1).strip()
                logger.log_event("FINAL_ANSWER", {"answer": final_answer, "steps": steps})
                return final_answer

            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()
                logger.log_event("TOOL_CALL", {"tool": tool_name, "input": tool_input})
                observation = self._execute_tool(tool_name, tool_input)
                logger.log_event("OBSERVATION", {"observation": observation})
                prompt = "Observation: " + observation
                self.history.append({"role": "user", "content": prompt})
            else:
                logger.log_event("PARSING_ERROR", {"step": steps, "response": response})
                prompt = "Could not parse your response. Use format: Thought: ... Action: tool_name[args] or Final Answer: ..."
                self.history.append({"role": "user", "content": prompt})

            steps += 1

        logger.log_event("AGENT_END", {"steps": steps})
        return "Max steps reached without final answer."

    def _execute_tool(self, tool_name, args):
        for tool in self.tools:
            if tool.get("name") == tool_name:
                tool_func = tool.get("func")
                if not callable(tool_func):
                    return f"Tool {tool_name} has no callable func."
                parsed_args = self._parse_tool_arguments(args)
                try:
                    if isinstance(parsed_args, (tuple, list)):
                        return str(tool_func(*parsed_args))
                    if parsed_args == "" or parsed_args is None:
                        return str(tool_func())
                    return str(tool_func(parsed_args))
                except Exception as exc:
                    return f"Error executing tool {tool_name}: {exc}"
        return f"Tool {tool_name} not found."

    def _parse_tool_arguments(self, args):
        if not args:
            return ""
        try:
            return ast.literal_eval(args.strip())
        except (ValueError, SyntaxError):
            return args.strip()
