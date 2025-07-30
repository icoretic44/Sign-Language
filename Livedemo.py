# live_demo.py (Đã sửa lỗi)
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque
import numpy as np

# --- 1. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Dropout(dropout_prob), nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x); return self.fc(out[:, -1, :])

# --- 2. CẤU HÌNH VÀ TẢI MÔ HÌNH ---
INPUT_SIZE = 126
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 5
DROPOUT_PROB = 0.5
MODEL_PATH = "bilstm_jsonl_best_model.pth"

CLASS_NAMES = {0: "a", 1: "b", 2: "c", 3:"d", 4:"thank_you"}

device = torch.device("cpu")
model = BiLSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT_PROB)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- 3. KHỞI TẠO MEDIAPIPE VÀ WEBCAM ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# --- 4. THIẾT LẬP BỘ ĐỆM VÀ VÒNG LẶP CHÍNH ---
SEQUENCE_LENGTH = 64
point_history = deque(maxlen=SEQUENCE_LENGTH)
predicted_label = "..."
confidence_threshold = 0.8

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # 1. Trích xuất tất cả các điểm tìm thấy vào một list phẳng
    all_landmarks_flat = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                all_landmarks_flat.extend([lm.x, lm.y, lm.z])

    # 2. Tạo một vector 0 có đúng kích thước INPUT_SIZE (126)
    current_frame_points = np.zeros(INPUT_SIZE)
    
    # 3. Điền các điểm đã tìm thấy vào đầu vector 0.
    #    Nếu chỉ có 1 tay (63 điểm), 63 vị trí đầu sẽ được điền, còn lại là 0.
    #    Nếu có 2 tay (126 điểm), toàn bộ vector sẽ được điền.
    #    Nếu không có tay, vector vẫn là 126 số 0.
    num_points_detected = len(all_landmarks_flat)
    current_frame_points[:num_points_detected] = all_landmarks_flat
    
    # Thêm vector 126 chiều này vào lịch sử
    point_history.append(current_frame_points)

    # --- 5. SUY LUẬN KHI CÓ ĐỦ DỮ LIỆU ---
    if len(point_history) == SEQUENCE_LENGTH:
        # Chuyển deque (list) của các numpy array thành một numpy array lớn duy nhất
        input_np_array = np.array(list(point_history))
        # Thêm chiều batch (1) vào phía trước
        input_np_array_batched = np.expand_dims(input_np_array, axis=0)
        # Tạo tensor từ numpy array
        input_tensor = torch.tensor(input_np_array_batched, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            if confidence.item() > confidence_threshold:
                predicted_label = f"{CLASS_NAMES[predicted_idx.item()]} ({confidence.item():.0%})"
            else:
                predicted_label = "..."
    
    cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Live Sign Language Classification', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()