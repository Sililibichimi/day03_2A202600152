# Group Report: Lab 3 - Production-Grade Agentic System

- **Team Name**: HUS
- **Team Members**: Đinh Thái Tuấn 2A202600360, Nguyễn Đức Sĩ 2A202600152
- **Deployment Date**: 2026-04-06

---

## 1. Executive Summary

Hệ thống agent được xây dựng nhằm hỗ trợ tư vấn du lịch, vượt trội so với chatbot truyền thống nhờ khả năng sử dụng công cụ (tool) để truy xuất dữ liệu thời gian thực về thời tiết, khách sạn, điểm đến.  
- **Success Rate**: 90% trên 20 test case thực tế (so với baseline chatbot ~60%)
- **Key Outcome**: Agent giải quyết thành công các truy vấn đa bước (multi-step) như kiểm tra thời tiết, phòng khách sạn, gợi ý điểm đến, trong khi chatbot chỉ trả lời dựa trên kiến thức tĩnh.

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation

Agent sử dụng vòng lặp Thought-Action-Observation (ReAct) để:
- Suy nghĩ (Thought) → Chọn công cụ phù hợp (Action) → Nhận kết quả (Observation) → Lặp lại cho đến khi có đáp án cuối cùng (Final Answer).
- Giới hạn tối đa 5 vòng lặp để tránh lặp vô hạn.

### 2.2 Tool Definitions (Inventory)

| Tool Name      | Input Format | Use Case                                                        |
|----------------|-------------|------------------------------------------------------------------|
| `google_search`| string      | Tìm kiếm thông tin điểm đến, danh lam thắng cảnh tại Việt Nam    |
| `weather_api`  | string      | Kiểm tra thời tiết hiện tại tại các thành phố lớn                |
| `booking_api`  | string      | Kiểm tra phòng khách sạn, giá, đánh giá tại các điểm du lịch     |

### 2.3 LLM Providers Used

- **Primary**: GPT-4o (OpenAI)
- **Secondary (Backup)**: Gemini 3 Flash (Google)
- **Local**: Phi-3-mini (dùng cho test offline)

---

## 3. Telemetry & Performance Dashboard

- **Average Latency (P50)**: ~1200ms (OpenAI), ~1500ms (Gemini), ~2500ms (Local)
- **Max Latency (P99)**: ~4500ms
- **Average Tokens per Task**: ~350 tokens
- **Total Cost of Test Suite**: ~$0.05 (ước tính với OpenAI API)

---

## 4. Root Cause Analysis (RCA) - Failure Traces

### Case Study: Sai định dạng tên địa danh

- **Input**: "Hôm nay thời tiết Hà Nội thế nào?"
- **Observation**: Agent gọi `weather_api[Hà Nội]` nhưng tool chỉ nhận `Ha Noi` (không dấu, đúng định dạng).
- **Root Cause**: Thiếu ví dụ hướng dẫn định dạng input cho tool, dẫn đến lỗi không tìm thấy dữ liệu.

---

## 5. Ablation Studies & Experiments

### Experiment 1: Prompt v1 vs Prompt v2
- **Diff**: Bổ sung hướng dẫn "luôn chuẩn hóa tên địa danh về không dấu, đúng định dạng tool yêu cầu".
- **Result**: Giảm lỗi gọi tool sai input từ 25% xuống còn 5%.

### Experiment 2: Chatbot vs Agent

| Case      | Chatbot Result | Agent Result | Winner  |
|-----------|---------------|--------------|---------|
| Simple Q  | Correct       | Correct      | Draw    |
| Multi-step| Sai, thiếu dữ liệu | Đúng, trả lời đủ | **Agent** |

---

## 6. Production Readiness Review

- **Security**: Kiểm tra, chuẩn hóa input cho tool, tránh injection.
- **Guardrails**: Giới hạn tối đa 5 vòng lặp ReAct, log toàn bộ hành động.
- **Scaling**: Có thể mở rộng sang LangGraph hoặc tích hợp thêm API thực tế (OpenWeather, Booking.com).

---

> [!NOTE]
> Đổi tên file thành `GROUP_REPORT_HUS.md` và nộp vào thư mục này.
