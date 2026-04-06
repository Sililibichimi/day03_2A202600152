# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyễn Đức Sĩ
- **Student ID**: 2A202600152
- **Date**: 2026-04-06

---

## I. Technical Contribution (15 Points)

Tôi chịu trách nhiệm chính về hai phần cốt lõi của hệ thống: **thiết kế system prompt** cho ReAct Agent và **thiết kế module tool chạy** (tool execution pipeline).

- **Modules Implemented**:
  - `src/agent/agent.py` — System prompt design & tool execution logic (`get_system_prompt`, `_execute_tool`, `_parse_tool_arguments`)
  - `src/tools/base.py` — Abstract base class `BaseTool` định nghĩa interface chung cho mọi tool
  - `src/tools/__init__.py` — Factory function `get_all_tools()` để đăng ký và khởi tạo toàn bộ tool inventory

- **Code Highlights**:

  **1. System Prompt Design** (`src/agent/agent.py:16-28`):
  ```python
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
  ```
  System prompt được thiết kế ngắn gọn nhưng đủ 3 thành phần quan trọng: (1) vai trò của agent, (2) danh sách công cụ kèm mô tả, (3) định dạng output bắt buộc theo pattern ReAct `Thought → Action → Observation → Final Answer`. Prompt được viết bằng tiếng Việt không dấu để tương thích tốt với cả OpenAI và Gemini.

  **2. Tool Execution Pipeline** (`src/agent/agent.py:66-81`):
  ```python
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
  ```
  Pipeline này thực hiện 3 bước: (1) tìm tool theo tên trong registry, (2) parse đối số từ string thành kiểu Python thích hợp qua `ast.literal_eval`, (3) gọi hàm và bắt lỗi — nếu lỗi thì trả về message mô tả thay vì crash toàn bộ vòng lặp ReAct.

  **3. Argument Parsing** (`src/agent/agent.py:83-89`):
  ```python
  def _parse_tool_arguments(self, args):
      if not args:
          return ""
      try:
          return ast.literal_eval(args.strip())
      except (ValueError, SyntaxError):
          return args.strip()
  ```
  Sử dụng `ast.literal_eval` để an toàn parse các kiểu phức tạp (tuple, dict, list) từ string, fallback về raw string nếu không parse được.

  **4. Base Tool Interface** (`src/tools/base.py`):
  ```python
  class BaseTool(ABC):
      def __init__(self, name: str, description: str):
          self.name = name
          self.description = description

      @abstractmethod
      def run(self, *args, **kwargs) -> str:
          pass

      def to_dict(self) -> Dict[str, Any]:
          return {
              "name": self.name,
              "description": self.description,
              "func": self.run,
          }
  ```
  Interface này cho phép mọi tool kế thừa đều có cùng format khi đăng ký vào agent, đảm bảo tính mở rộng — chỉ cần tạo class mới kế thừa `BaseTool` là agent tự nhận diện được.

- **Documentation**:
  System prompt định hướng LLM sinh output theo đúng format `Action: tool_name[args]`, sau đó regex parser trong `run()` (`agent.py:40-41`) trích xuất tên tool và đối số. Kết quả được chuyển vào `_execute_tool()` để gọi hàm thực tế, observation trả về được append vào history và tiếp tục vòng lặp cho đến khi LLM sinh `Final Answer`.

---

## II. Debugging Case Study (10 Points)

### Case 1: Tool input format mismatch — diacritics cause failure cascade

- **Problem Description**: Với GPT-4o, agent gọi `weather_api[Hà Nội]` nhưng tool chỉ nhận key `"ha noi"` (không dấu, lowercase). Kết quả: 5 lần gọi tool liên tiếp thất bại, agent đạt `max_steps=5` mà không có Final Answer.

- **Log Source**: `logs/2026-04-06.log`, dòng 78-96:
  ```
  Step 0: weather_api[Hà Nội] → "Weather data not available for 'Hà Nội'."
  Step 1: weather_api[Hanoi]  → "Weather data not available for 'Hanoi'."
  Step 2: booking_api[Hà Nội] → "Booking data not available for 'Hà Nội'."
  Step 3: google_search["thời tiết Hà Nội hôm nay"] → "Location not found"
  Step 4: google_search["thời tiết Ha Noi hôm nay"] → "Location not found"
  → AGENT_END: steps=5, total_tokens=1776, total_latency_ms=10889, tool_calls=5
  ```

- **Diagnosis**: Tool database dùng lowercase không dấu (`"ha noi"`) nhưng LLM sinh input có dấu (`"Hà Nội"`, `"Hanoi"`). System prompt không hướng dẫn cách chuẩn hóa tên địa danh. Agent cố gắng retry nhưng mỗi lần vẫn sai format → lãng phí 1776 tokens và ~11s mà không có kết quả.

- **Solution**: 
  - Cải tiến tool `run()` để normalize input: `location.lower().strip()` và map các biến thể tên (đã được implement trong `travel_tools.py`).
  - Cải tiến system prompt: thêm hướng dẫn "luôn dùng tên địa danh không dấu, viết hoa chữ cái đầu mỗi từ (ví dụ: 'Ha Noi', 'Da Nang', 'Ho Chi Minh')".
  - Với case query 2 ("Da Nang"), agent hoạt động đúng vì input đã khớp format → thành công ở step 4 với 2090 tokens, 4 tool calls.

### Case 2: Gemini hallucinates Observation (không gọi tool thật)

- **Problem Description**: Gemini 3.1 Flash Lite tự sinh cả Action + Observation + Final Answer trong 1 response (step=0), không gọi tool thật.
- **Log Source**: `logs/2026-04-06.log`, dòng 21-22
- **Diagnosis**: Model nhỏ có xu hướng "điền vào chỗ trống" thay vì chờ system gọi tool.
- **Solution**: Cần thêm chỉ dẫn trong prompt: "KHÔNG tự sinh Observation — Observation là kết quả do hệ thống trả về sau khi gọi tool."

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: Block `Thought` giúp agent "nghĩ trước khi hành động". Dữ liệu thực tế từ metrics cho thấy:
   - **Chatbot** (GPT-4o): 1 request, 898 tokens, 8332ms → trả lời chung chung "tôi không có dữ liệu thời gian thực"
   - **Agent** (GPT-4o, query Da Nang): 5 requests, 2090 tokens, 9648ms, 4 tool calls → trả lời có dữ liệu cụ thể (nhiệt độ 28°C, giá $45/đêm, rating 4.6/5)
   - Agent tốn ~2.3x tokens nhưng cho câu trả lời chính xác và có dữ liệu thực.

2. **Reliability**: Agent hoạt động **tệ hơn** chatbot khi:
   - Tool input format không khớp (case Hà Nội ở trên) — agent waste 5 steps, chatbot ít nhất còn trả lời được dù chung chung.
   - Với Gemini Flash models, agent hallucinate Observation → kết quả trông đúng nhưng thực chất là bịa, nguy hiểm hơn là chatbot nói thẳng "tôi không có dữ liệu".

3. **Observation**: Từ log metrics, mỗi tool call thất bại tốn trung bình ~1800ms latency và ~200 tokens. Khi observation trả về "not found", agent có thể retry nhưng nếu không được hướng dẫn đúng format sẽ lặp lại lỗi. Đây là điểm then chốt: **chất lượng observation quyết định hiệu quả của toàn bộ ReAct loop**.

### Session Metrics Summary (GPT-4o, 2 queries)

| Metric | Value |
|---|---|
| Total Requests | 12 |
| Total Tokens | 5,790 |
| Avg Latency | 3,206ms |
| P50 Latency | 1,857ms |
| P99 Latency | 9,607ms |
| Total Cost | $0.029 |
| Chatbot tokens/query | ~898-1,026 |
| Agent tokens/query (success) | ~2,090 |
| Agent tokens/query (fail) | ~1,776 (no answer) |

---

## IV. Future Improvements (5 Points)

- **Scalability**: Hiện tại tool registry là danh sách tuyến tính O(n) — chuyển sang dictionary lookup O(1). Với hệ thống nhiều tool (50+), dùng vector embedding để agent tự chọn tool phù hợp nhất với ngữ cảnh trước khi gọi.
- **Safety**: Thêm input validation (kiểm tra injection, độ dài, ký tự đặc biệt). Implement retry limit per tool — nếu tool thất bại 2 lần liên tiếp, agent nên chuyển sang tool khác hoặc trả lời "không có dữ liệu" thay vì retry vô hạn.
- **Performance**: Từ metrics thực tế, P99 latency lên tới ~9.6s (do chatbot GPT-4o response dài 611 tokens). Cần implement: (1) Cache kết quả tool call trùng lặp, (2) Streaming response để giảm perceived latency, (3) Timeout per step (ví dụ: 5s) để tránh agent bị treo.
- **Prompt Engineering**: Chuyển system prompt sang dạng structured (JSON schema) thay vì plain text. Thêm few-shot examples cho từng loại query và từng tool. Đặc biệt: hướng dẫn rõ format input cho từng tool để tránh case mismatch như Hà Nội/Ha Noi/ha noi.

---
