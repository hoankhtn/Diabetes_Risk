# Hai câu hỏi phân tích không sử dụng học máy

## Dữ liệu và phạm vi

- Nguồn trong dự án: `data/raw/diabetes_risk.csv`, gồm 15.000 hồ sơ.
- `diabetes_risk` có ba nhãn: `Low` (thấp), `Moderate` (trung bình), `High` (cao).
- Đây là dữ liệu tổng hợp. Trong tài liệu này, “nguy cơ cao” chỉ nhãn `High` của dữ liệu, không phải chẩn đoán hoặc xác suất mắc bệnh thực tế.
- Dữ liệu không có ngày khám hay lịch sử theo dõi. Câu hỏi 2 dùng pivot để đáp ứng phần reshaping của D2, không tạo thêm thời gian giả.
- Tài liệu đề xuất câu hỏi và cách phân tích; chưa trình bày kết quả trả lời hai câu hỏi.

## Câu hỏi 1 — Ngủ ít kết hợp với BMI cao (D1 + D3)

> Người vừa ngủ ít vừa có BMI cao có tỷ lệ thuộc nhóm nguy cơ cao khác thế nào so với người chỉ có một hoặc không có hai đặc điểm này?

BMI là chỉ số khối cơ thể. Trong câu hỏi này, “BMI cao” được hiểu là cao hơn mức trung vị của người cùng nhóm tuổi và cùng giới trong mẫu, không phải phân loại béo phì theo ngưỡng y khoa.

### Motivation

- **Tại sao đáng trả lời?** Nhìn riêng giờ ngủ hoặc BMI chưa cho thấy hai đặc điểm cùng xuất hiện có đi kèm một mức chênh lệch nguy cơ khác hay không. So sánh bốn nhóm giúp tìm hiểu liệu mối liên hệ giữa ngủ ít và nhãn nguy cơ có thay đổi tùy mức BMI. Câu trả lời vẫn có giá trị nếu không phát hiện khác biệt.
- **Ai quan tâm?** Nhóm nghiên cứu lối sống, nhóm xây dựng nội dung giáo dục sức khỏe và người phân tích bộ dữ liệu.
- **Phục vụ quyết định gì?** Giúp chọn giả thuyết và nhóm đặc điểm cần khảo sát thêm: giờ ngủ riêng lẻ hay sự kết hợp giữa giờ ngủ và BMI. Với dữ liệu mô phỏng, kết quả hỗ trợ lựa chọn hướng nghiên cứu và ví dụ minh họa, chưa đủ để lựa chọn can thiệp cho người thật.

## Câu hỏi 2 — Thu nhập và vai trò của tuổi (D2 + D4)

> Chênh lệch tỷ lệ nguy cơ cao giữa các mức thu nhập có còn giữ nguyên khi so sánh những người trong cùng nhóm tuổi?

### Motivation

- **Tại sao đáng trả lời?** Hai nhóm thu nhập có thể khác nhau về tỷ lệ nguy cơ cao vì cơ cấu tuổi khác nhau. Nếu chỉ so sánh tỷ lệ chung, người đọc có thể gán sự khác biệt cho thu nhập dù một phần liên quan đến tuổi. Cần kiểm tra cả trong từng nhóm tuổi và sau khi đưa các mức thu nhập về cùng cơ cấu tuổi.
- **Ai quan tâm?** Nhóm nghiên cứu bất bình đẳng sức khỏe, người lập kế hoạch khảo sát và nhóm phân tích dữ liệu.
- **Phục vụ quyết định gì?** Giúp quyết định có cần phân tầng hoặc cân bằng tuổi khi thiết kế khảo sát tiếp theo, và có nên tiếp tục nghiên cứu riêng mối liên hệ giữa thu nhập và nhãn nguy cơ. Kết quả cũng giúp chọn cách trình bày so sánh công bằng hơn giữa các mức thu nhập.

## Đối chiếu yêu cầu

| Câu hỏi | Tiêu chí | Bằng chứng cần có khi thực hiện |
| --- | --- | --- |
| 1 | D1 — Kết hợp bảng | Bảng tham chiếu tuổi–giới, ghép nhiều–một, số dòng trước/sau và khóa không khớp |
| 1 | D3 — Biến dẫn xuất | Độ lệch BMI so với nhóm tham chiếu và bốn nhóm kết hợp BMI–giờ ngủ |
| 2 | D2 — Reshaping | Bảng đếm dài được pivot thành ba cột nhãn nguy cơ để phân tích cơ cấu |
| 2 | D4 — Kiểm soát yếu tố gây nhiễu | So sánh trong từng nhóm tuổi và chuẩn hóa về cùng cơ cấu tuổi |

