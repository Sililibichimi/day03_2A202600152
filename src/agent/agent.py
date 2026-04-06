import ast
import re
from typing import List, Dict, Any, Optional, Tuple
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgent:
    def __init__(self, llm, tools, max_steps=5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []
        self.total_tokens = 0
        self.total_latency_ms = 0
        self.tool_calls = 0

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
        self.total_tokens = 0
        self.total_latency_ms = 0
        self.tool_calls = 0
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        prompt = user_input
        steps = 0
        while steps < self.max_steps:
            result = self.llm.generate_response(self.get_system_prompt(), self.history, prompt)
            response = result["content"]
            self.history.append({"role": "assistant", "content": response})

            # Track per-step metrics
            usage = result.get("usage", {})
            latency = result.get("latency_ms", 0)
            provider = result.get("provider", "unknown")
            tracker.track_request(provider, self.llm.model_name, usage, latency)

            self.total_tokens += usage.get("total_tokens", 0)
            self.total_latency_ms += latency

            logger.log_event("AGENT_STEP", {"step": steps, "response": response, "tokens": usage.get("total_tokens", 0), "latency_ms": latency})

            action_match = re.search(r"Action:\s*(\w+)\[([^\]]*)\]", response)
            final_match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)

            if final_match:
                final_answer = final_match.group(1).strip()
                logger.log_event("FINAL_ANSWER", {"answer": final_answer, "steps": steps, "total_tokens": self.total_tokens, "total_latency_ms": self.total_latency_ms, "tool_calls": self.tool_calls})
                return final_answer

            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()
                self.tool_calls += 1
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

        logger.log_event("AGENT_END", {"steps": steps, "total_tokens": self.total_tokens, "total_latency_ms": self.total_latency_ms, "tool_calls": self.tool_calls})
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
