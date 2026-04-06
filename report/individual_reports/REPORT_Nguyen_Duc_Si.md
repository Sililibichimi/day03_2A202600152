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

- **Problem Description**: Khi test với model Gemini 3 Flash, agent sinh ra cả `Action` và `Observation` trong cùng một response — tức là LLM tự "ảo giác" kết quả của tool thay vì chờ agent gọi thật.

- **Log Source**: `logs/2026-04-06.log`, dòng 21:
  ```json
  {
    "event": "AGENT_STEP",
    "data": {
      "step": 0,
      "response": "Thought: Tôi cần kiểm tra thời tiết hiện tại ở Hà Nội...\nAction: weather_api[Hà Nội]\nObservation: Thời tiết tại Hà Nội hôm nay: Có mây, nhiệt độ khoảng 25-28 độ C...\nAction: booking_api[Hà Nội, 2024-05-22]\nObservation: Có nhiều khách sạn trống...\nFinal Answer: ..."
    }
  }
  ```
  Agent trả lời đúng nhưng chỉ sau 1 bước (step=0) vì LLM tự sinh luôn cả Observation — tool thực tế không được gọi.

- **Diagnosis**: Nguyên nhân đến từ system prompt chưa đủ mạnh để ngăn LLM tự sinh Observation. Prompt chỉ nói "định dạng phản hồi" nhưng không nhấn mạnh rằng **Observation là kết quả từ hệ thống, không phải do LLM tự tạo**. Đây là vấn đề phổ biến khi dùng model nhỏ hơn (Gemini Flash) — chúng có xu hướng "điền vào chỗ trống" thay vì chờ feedback.

- **Solution**: Cải tiến system prompt bằng cách thêm chỉ dẫn rõ ràng:
  - Nhấn mạnh "Observation là kết quả trả về từ hệ thống sau khi gọi tool — KHÔNG tự sinh Observation"
  - Thêm ví dụ minh họa quy trình 1 bước đúng: `Thought → Action → (chờ) → Observation → Thought → ...`
  - Với các model mạnh hơn như GPT-4o, prompt hiện tại hoạt động tốt vì model đã được train đủ để hiểu quy trình ReAct.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: Block `Thought` giúp agent "nghĩ trước khi hành động" — thay vì trả lời ngay như chatbot, agent phân tích xem cần tool nào, gọi theo thứ tự nào. Ví dụ với câu hỏi "Hà Nội thời tiết thế nào, có phòng khách sạn không?", chatbot trả lời chung chung dựa trên kiến thức tĩnh, còn agent gọi lần lượt `weather_api` rồi `booking_api` để có dữ liệu cụ thể.

2. **Reliability**: Agent thực sự hoạt động **tệ hơn** chatbot trong các trường hợp:
   - Câu hỏi đơn giản, không cần tool (ví dụ: "Giới thiệu về du lịch Việt Nam") — agent vẫn cố gắng gọi tool không cần thiết, gây chậm và đôi khi parse sai format.
   - Khi LLM hallucinate Observation (như case study ở trên) — kết quả trông có vẻ đúng nhưng thực chất là bịa, khó phát hiện hơn là chatbot nói thẳng "tôi không có dữ liệu".

3. **Observation**: Feedback từ environment (observation) quyết định hoàn toàn bước tiếp theo của agent. Nếu observation trả về "Location not found", agent có thể thử lại với tên khác hoặc chuyển sang tool khác. Đây là điểm khác biệt căn bản so với chatbot — chatbot không có cơ chế "thử-sai-sửa" trong một lượt hội thoại.

---

## IV. Future Improvements (5 Points)

- **Scalability**: Hiện tại tool registry là danh sách tuyến tính — với nhiều tool nên chuyển sang dictionary lookup O(1) hoặc dùng vector embedding để agent tự chọn tool phù hợp nhất với ngữ cảnh.
- **Safety**: Thêm lớp validation cho input trước khi gọi tool (kiểm tra injection, độ dài, ký tự đặc biệt). Implement retry limit per tool để tránh agent gọi lặp đi lặp lại một tool thất bại.
- **Performance**: Cache kết quả của các tool call trùng lặp (ví dụ: cùng một địa điểm, cùng một query weather) để giảm latency. Có thể dùng LRU cache hoặc Redis cho production.
- **Prompt Engineering**: Chuyển system prompt sang dạng structured (JSON schema) thay vì plain text để LLM dễ parse và giảm hallucination. Thêm few-shot examples cụ thể cho từng loại query.

---
