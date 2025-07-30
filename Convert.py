# converter.py
import os
import json
from tqdm import tqdm

SOURCE_DIR = r"G:\My Drive\Data_p1" # 

OUTPUT_DIR = r"G:\My Drive\AI4LI_Dataset_JSONL" 

# ==============================================================================

def convert_to_jsonl(source_root, output_root):
    """
    Quét qua cấu trúc thư mục cũ, chuyển đổi mỗi session thành một file .jsonl duy nhất
    và lưu vào thư mục đầu ra.
    """
    print(f"Bắt đầu quá trình chuyển đổi từ: '{source_root}'")
    print(f"Dữ liệu sẽ được lưu tại: '{output_root}'")
    
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_root, exist_ok=True)
    
    # Đếm tổng số session để tạo thanh tiến trình chính xác
    total_sessions = sum([len(os.listdir(os.path.join(source_root, label))) for label in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, label))])
    
    with tqdm(total=total_sessions, desc="Tổng tiến độ") as pbar:
        # Duyệt qua từng thư mục label (ví dụ: 'a', 'b')
        for label in os.listdir(source_root):
            source_label_path = os.path.join(source_root, label)
            if not os.path.isdir(source_label_path):
                continue

            # Tạo thư mục label tương ứng trong thư mục đầu ra
            output_label_path = os.path.join(output_root, label)
            os.makedirs(output_label_path, exist_ok=True)

            # Duyệt qua từng thư mục session (ví dụ: 'a_01', 'a_02')
            for session_folder in os.listdir(source_label_path):
                source_session_path = os.path.join(source_label_path, session_folder)
                coords_path = os.path.join(source_session_path, "coords")
                
                if not os.path.isdir(coords_path):
                    continue
                
                # Lấy danh sách các file JSON và sắp xếp theo đúng thứ tự frame
                json_files = [f for f in os.listdir(coords_path) if f.endswith('.json')]
                try:
                    sorted_json_files = sorted(json_files, key=lambda fn: int(fn.split('_')[0]))
                except (ValueError, IndexError):
                    print(f"\nCảnh báo: Không thể sắp xếp file trong thư mục {coords_path}. Bỏ qua...")
                    pbar.update(1)
                    continue

                if not sorted_json_files:
                    pbar.update(1)
                    continue
                    
                # Tạo tên file .jsonl mới
                output_jsonl_filename = f"{session_folder}.jsonl"
                output_jsonl_path = os.path.join(output_label_path, output_jsonl_filename)
                
                # Mở file .jsonl để ghi
                with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
                    # Đọc từng file JSON, chuyển thành chuỗi và ghi vào file .jsonl
                    for json_file in sorted_json_files:
                        file_path = os.path.join(coords_path, json_file)
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            data = json.load(infile)
                            # Ghi chuỗi JSON và thêm ký tự xuống dòng
                            outfile.write(json.dumps(data) + '\n')
                
                pbar.update(1) # Cập nhật thanh tiến trình

if __name__ == "__main__":
    # Kiểm tra xem thư mục nguồn có tồn tại không
    if not os.path.isdir(SOURCE_DIR):
        print(f"LỖI: Không tìm thấy thư mục nguồn '{SOURCE_DIR}'.")
        print("Vui lòng kiểm tra lại đường dẫn trong biến SOURCE_DIR.")
    else:
        convert_to_jsonl(SOURCE_DIR, OUTPUT_DIR)
        print("\nChuyển đổi hoàn tất!")
        print(f"Toàn bộ dữ liệu đã được chuyển sang định dạng JSONL và lưu tại: '{OUTPUT_DIR}'")