# Theo dõi và làm sạch dữ liệu (Mục 3.4 & D1)

Các hàm trả về bản sao, giữ nguyên dữ liệu đầu vào. Chạy ví dụ từ thư mục gốc
dự án; trong notebook ở `notebooks/`, thêm thư mục gốc vào `sys.path` trước khi
import `src`.

## Quy trình cần mô tả trước phần code trong mỗi câu hỏi

1. Chọn cột cần phân tích và kiểm tra cách mã hóa giá trị thiếu. Chỉ đổi `-1`,
   `999` thành thiếu khi xác nhận chúng là placeholder của cột đó. Không chọn
   `patient_id`: mã bệnh nhân `999` là hợp lệ trong bộ dữ liệu này.
2. Chuyển cột đã chọn sang số. Chuỗi không chuyển được và ±∞ trở thành thiếu;
   thay placeholder bằng NaN. Hai bước này không xóa dòng và đều báo số ô đổi,
   cùng số dòng trước/sau.
3. Chọn chính sách cho giá trị thiếu: mặc định giữ lại. Nếu loại các dòng thiếu
   ở cột đang phân tích, ghi rõ cột và báo số dòng bị loại.
4. Rà soát ngoại lai bằng IQR hoặc z-score. Mặc định chỉ làm sạch giá trị, không
   lọc ngoại lai. Có thể lưu cờ để kiểm tra độ nhạy rồi mới quyết định lọc.
   Nếu lọc, loại dòng có ít nhất một cờ và báo số dòng duy nhất bị loại,
   không cộng số cờ của nhiều cột.
5. Nếu ghép bảng tham chiếu, nêu khóa, kiểu join và quan hệ dự kiến (ví dụ nhiều
   bệnh nhân–một nhóm tuổi/giới). Kiểm tra khóa thiếu, khóa lặp, khóa không khớp
   ở cả hai bảng và số dòng sau ghép. Điều tra khóa không khớp trước khi dùng
   kết quả phân tích; left join giữ số dòng không đồng nghĩa với khớp đủ khóa.

Mô tả trên là mẫu quy trình: điều chỉnh theo câu hỏi, nêu lý do chọn và chỉ
thực hiện bước cần thiết. Không cần xóa ngoại lai chỉ để có bước lọc.

## Làm sạch cột số và ngoại lai

```python
import pandas as pd
from src.preprocessing import clean_numeric_columns, detect_outliers, filter_outliers
from src.utils import track_rows, verify_merge

df = pd.read_csv("data/raw/diabetes_risk.csv")
numeric_columns = ["bmi", "hours_sleep_per_night"]
clean = clean_numeric_columns(df, numeric_columns)

# Giữ cờ để rà soát trước; thao tác này không xóa dòng.
iqr_flags = detect_outliers(clean, numeric_columns, method="iqr")
z_flags = detect_outliers(clean, numeric_columns, method="zscore")

# Ví dụ phân tích độ nhạy, chỉ dùng nếu có lý do loại ngoại lai.
sensitivity = filter_outliers(clean, numeric_columns, method="iqr")

# Hoặc gọi cả quy trình, có báo cáo riêng sau từng bước.
complete_cases = clean_numeric_columns(
    df, numeric_columns, drop_missing=True, outlier_method="zscore",
    z_threshold=3.0,
)
```

Các hàm xử lý từng phần là `replace_placeholders`, `detect_outliers` và
`filter_outliers`. `clean_numeric_columns` gọi chúng theo thứ tự: chuyển kiểu
→ thay placeholder → tùy chọn bỏ thiếu → tùy chọn lọc ngoại lai. Nếu các cột có
placeholder khác nhau, gọi riêng cho từng nhóm cột với `placeholders=[...]`;
truyền `placeholders=[]` nếu không có mã thiếu cần thay.

- **IQR:** nằm ngoài `[Q1 − k×IQR, Q3 + k×IQR]`, mặc định `k=1.5`.
  Phân vị nội suy tuyến tính; hai hàng rào làm tròn 10 chữ số thập phân.
- **Z-score:** `abs((x − mean)/std) > z_threshold`, mặc định ngưỡng 3,
  độ lệch chuẩn tổng thể (`ddof=0`).
- Giá trị đúng bằng ngưỡng được giữ. NaN không bị gắn cờ; chỉ bị loại khi
  `drop_missing=True`. Hàm phát hiện/lọc trực tiếp yêu cầu cột số hữu hạn hoặc
  NaN; dùng `clean_numeric_columns` trước nếu dữ liệu có chuỗi hay ±∞.
- Cột toàn thiếu, IQR bằng 0 hoặc độ lệch chuẩn bằng 0 không phát sinh cờ theo
  phương pháp tương ứng. Đây là quy ước nhất quán với notebook khám phá, cần
  rà soát riêng các cột suy biến nếu chúng có ý nghĩa với câu hỏi.
- Ngưỡng được tính một lần trên dữ liệu đưa vào, từng cột độc lập. IQR ít phụ
  thuộc mean/SD hơn z-score; z-score dễ chịu ảnh hưởng bởi phân phối lệch và
  điểm cực trị. Cả hai chỉ là quy tắc thống kê, không chứng minh số đo sai.

Với ML, chia train/test trước và chỉ ước lượng tham số trên train. Các hàm lọc
này phục vụ phân tích mô tả/độ nhạy, không phải transformer lưu ngưỡng đã fit.

## Kiểm tra dòng và gộp bảng

```python
before = clean
after = before.dropna(subset=["bmi"])
row_report = track_rows(
    before, after, "Loại dòng thiếu BMI",
    reason="Cần BMI để tính trung vị nhóm tham chiếu.",
)

# Ví dụ khóa tuổi–giới; thay age bằng nhóm tuổi khi phân tích có phân nhóm.
reference = (
    after.groupby(["age", "gender"], as_index=False)["bmi"]
    .median().rename(columns={"bmi": "reference_bmi"})
)
track_rows(after, reference, "Tạo bảng tham chiếu",
           reason="Mỗi tổ hợp tuổi–giới trở thành một dòng trung vị BMI.")
joined, merge_report = verify_merge(
    clean, reference, on=["age", "gender"], how="left",
    validate="many_to_one", step_name="Ghép BMI tham chiếu",
)
display(merge_report["left_only_keys"])
display(merge_report["right_only_keys"])
```

`track_rows(df_before, df_after, step_name)` in và trả về dictionary với
`rows_before`, `rows_after`, `change`, `change_pct`, `reason`. `change` là số
dòng sau trừ trước; nếu đầu vào rỗng thì tỷ lệ là `None` và hiển thị `N/A`.
Tên bước hoặc `reason` cần giải thích phép biến đổi: số dòng không thể tự
chứng minh nguyên nhân hoặc phát hiện việc xóa rồi thêm lại cùng số dòng.

`verify_merge` thực hiện phép ghép và trả về `(joined, merge_report)`:

- `left_only_keys`, `right_only_keys`: toàn bộ khóa duy nhất không khớp;
  màn hình chỉ in tối đa 5 ví dụ mỗi phía. Kiểm tra trước khi ghép nên vẫn thấy
  khóa bị mất qua `inner join`.
- `unmatched_left_rows`, `unmatched_right_rows`: số dòng đầu vào tương ứng,
  tính cả nhiều dòng dùng chung một khóa không khớp.
- `matched_keys`, `duplicate_left_rows`, `duplicate_right_rows`: số khóa khớp
  và số dòng thuộc khóa lặp ở mỗi bảng (tính cả lần xuất hiện đầu tiên).
- Số dòng hai bảng đầu vào, số dòng kết quả và chênh lệch so với bảng trái.
- Giải thích chênh lệch bằng số dòng trái bị loại (`excluded_left_rows`), số
  dòng chỉ có bên phải được thêm (`added_right_rows`) và số dòng tăng do khóa
  khớp nhiều lần (`matched_expansion_rows`).

Mặc định `validate="many_to_one"` gây `pandas.errors.MergeError` nếu bảng bên phải
có khóa lặp. Chọn `one_to_one`, `one_to_many` hoặc `many_to_many` khi đúng với
quan hệ thực tế. Khóa thiếu gây `ValueError`; nếu chủ động chọn
`allow_null_keys=True`, pandas có thể ghép khóa null với nhau. Chế độ này báo
thêm `null_left_rows`/`null_right_rows`, cần cân nhắc trước khi diễn giải.

Các hàm chỉ tự báo khi được gọi. Mỗi thao tác lọc, drop hoặc gộp khác trong
notebook vẫn cần gọi `track_rows` ngay sau thao tác. Tài liệu này cung cấp
tiền xử lý; phần phương pháp, ít nhất hai biểu đồ có nhãn, kết quả định lượng
và giới hạn vẫn cần viết riêng cho từng câu hỏi theo Mục 3.4.

## Kiểm tra

```console
python -m unittest discover -s tests -v
```
