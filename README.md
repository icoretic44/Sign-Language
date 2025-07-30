## ✋ Gesture Recognition with biLSTM
A simple pipeline for gesture recognition using MediaPipe and a trained biLSTM model. This project includes data generation, preprocessing, live demonstration, and recovery video creation.

## 📦 Project Structure :
```bash
├── requirements.txt       # Python dependencies
├── bilstm_jsonl_best_model.pth              # Pretrained biLSTM model
├── MediaPipe.py           # Generate hand gesture data using MediaPipe
├── Convert.py             # Convert generated data to optimized JSONL format
├── Recovery.py            # Create short gesture recovery videos
├── Livedemo.py            # Run live demo with trained biLSTM model
```
## 📚 Installation :
pip install -r requirements.txt

## 🚀 Usage :
1. Generate Data from Webcam (MediaPipe)
```bash
python MediaPipe.py
```
This script uses MediaPipe to record hand gestures and save keypoint data.

2. Convert Data Format (for Training)
```bash
python Convert.py
```
This will optimize data and convert it into JSONL format for training with Google Colab.

3. Live Demo with Pretrained biLSTM
```bash
python Livedemo.py
```
Uses bilstm_jsonl_best_model.pth to perform real-time gesture recognition from webcam input.

4. Create Recovery Videos
```bash
python Recover.py --session {directory to a file needed to recover}
```
## 🧠 Model Info :
Model: biLSTM

File: bilstm_jsonl_best_model.pth

Input: Hand keypoints (via MediaPipe)

Output: Gesture class label


