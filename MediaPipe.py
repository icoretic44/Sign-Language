import os
import json
import cv2
import mediapipe as mp
from datetime import datetime

class MediaPipeHandsRecorder:
    def __init__(self, base_dir=None, target_frames=64):
        # Thư mục gốc chứa mọi label
        self.base_dir      = base_dir or r"G:\My Drive\Data_p1"
        os.makedirs(self.base_dir, exist_ok=True)

        self.target_frames = target_frames

        # Cấu hình MediaPipe Hands
        self.mp_hands      = mp.solutions.hands
        self.drawing       = mp.solutions.drawing_utils
        self.styles        = mp.solutions.drawing_styles
        self.detector      = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # trạng thái
        self.recording     = False
        self.frame_counter = 0
        self.session_dir   = None
        self.coords_dir    = None

    def detect_and_draw(self, frame):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    frame, lm, self.mp_hands.HAND_CONNECTIONS,
                    self.styles.get_default_hand_landmarks_style(),
                    self.styles.get_default_hand_connections_style()
                )
        return frame, results

    def start_session(self):
        """Tạo folder tạm session_<timestamp> chỉ chứa coords"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.base_dir, f"session_{ts}")
        self.coords_dir  = os.path.join(self.session_dir, "coords")
        os.makedirs(self.coords_dir, exist_ok=True)

        self.recording     = True
        self.frame_counter = 0
        print(f"[INFO] Bắt đầu ghi {self.target_frames} frame metadata vào:\n  {self.coords_dir}")

    def save_frame(self, results):
        """Chỉ lưu JSON metadata về landmarks và handedness"""
        data = {
            "frame_index": self.frame_counter,
            "hands": []
        }

        hands_lms = results.multi_hand_landmarks or []
        hands_hd  = results.multi_handedness    or []

        for lms, hd in zip(hands_lms, hands_hd):
            hand = {
                "handedness": {
                    "label":  hd.classification[0].label,
                    "score":  float(hd.classification[0].score)
                },
                "landmarks": [
                    {"id": i, "x": lm.x, "y": lm.y, "z": lm.z}
                    for i, lm in enumerate(lms.landmark)
                ]
            }
            data["hands"].append(hand)

        json_path = os.path.join(self.coords_dir, f"{self.frame_counter:03d}_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.frame_counter += 1

    def finalize_session(self):
        """Hỏi label, đổi tên thư mục coords theo label và chỉ số"""
        label = input("Nhập tên ký hiệu (label): ").strip()
        if not label:
            print("[WARN] Label trống, giữ nguyên folder tạm.")
            return

        label_dir = os.path.join(self.base_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        existing = [
            d for d in os.listdir(label_dir)
            if os.path.isdir(os.path.join(label_dir, d))
            and d.startswith(label + "_")
        ]
        idx      = len(existing) + 1
        new_name = f"{label}_{idx:02d}"
        new_path = os.path.join(label_dir, new_name)

        os.rename(self.session_dir, new_path)
        print(f"[INFO] Đã chuyển '{self.session_dir}' → '{new_path}'")

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[ERROR] Không mở được nguồn video.")
            return

        print("[INFO] Nhấn 'z' để bắt đầu ghi metadata (JSON). Nhấn 'q' hoặc ESC để thoát.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Flip horizontally
            annotated, results = self.detect_and_draw(frame)
            cv2.imshow("Recorder", annotated)


            key = cv2.waitKey(1) & 0xFF
            # Bắt đầu session khi nhấn 'z'
            if key == ord('z') and not self.recording:
                self.start_session()

            # Nếu đang ghi, chỉ lưu JSON và kiểm tra giới hạn target_frames
            if self.recording:
                self.save_frame(results)
                if self.frame_counter >= self.target_frames:
                    print("[INFO] Đã ghi đủ metadata, dừng ghi.")
                    break

            if key in (ord('q'), 27):
                print("[INFO] Dừng chương trình.")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()

        if self.recording:
            self.finalize_session()
        print("[INFO] Kết thúc.")


if __name__ == "__main__":
    recorder = MediaPipeHandsRecorder()
    recorder.run(0)
