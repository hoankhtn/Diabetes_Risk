from pathlib import Path
import pandas as pd

#các cột trong file csv(ngoại trừ cột patient_id)
CATEGORICAL_COLUMNS = [
    "gender",
    "city",
    "family_history_diabetes",
    "physical_activity_level",
    "diet_type",
    "smoking_status",
    "alcohol_consumption",
    "income_bracket",
    "diabetes_risk"
]

#các cột bị thiếu dữ liệu
MISSING_COLUMNS = [
    "smoking_status",
    "alcohol_consumption",
    "income_bracket"
]

def Read_File(filename):
    LINKFILE = None

    for folder in [Path.cwd(), *Path.cwd().parents]:
        candidate = folder / "data" / "raw" / filename

        if candidate.is_file():
            LINKFILE = candidate
            break

    if LINKFILE is None:
        raise FileNotFoundError(
            f"Không tìm thấy {filename}. Thư mục hiện tại: {Path.cwd()}"
        )

    return pd.read_csv(LINKFILE)

def normalize_category_labels(df, columns):
    result = df.copy()

    for col in columns:
        values = result[col].astype("string").str.strip()

        # Chuyển chuỗi rỗng thành giá trị thiếu
        values = values.mask(values.eq(""))

        result[col] = values

    return result

def fill_missing_categories(df, columns):
    result = df.copy()

    for col in columns:
        result[col] = result[col].fillna("Unknown")

    return result

def preprocess_data(df):
    df = normalize_category_labels(
        df,
        CATEGORICAL_COLUMNS
    )

    df = fill_missing_categories(
        df,
        MISSING_COLUMNS
    )

    return df

def export_clean_file(df):
    # Lấy thư mục gốc của project
    project_root = Path(__file__).resolve().parents[1]

    # Đường dẫn đến thư mục processed
    output_path = project_root / "data" / "processed"

    # Tạo thư mục nếu chưa tồn tại
    output_path.mkdir(parents=True, exist_ok=True)

    # Xuất file CSV
    file_path = output_path / "diabetes_risk.csv"

    df.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Đã xuất file tại:", file_path)
    
if __name__ == "__main__":
    df = Read_File('diabetes_risk.csv')
    clean_df = preprocess_data(df)
    export_clean_file(clean_df)