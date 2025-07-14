# 🏀 Cross-Camera Player Re-Identification using YOLOv11

This project solves the problem of **player re-identification** across different camera views in sports footage using **YOLOv11**, an object detection model. It is designed for tasks such as analytics and broadcast tracking, where maintaining consistent player IDs across multiple views is essential.

---

## 📌 Problem Statement

> Given two different camera feeds (`broadcast.mp4` and `tacticam.mp4`) capturing the same sports scene, the task is to assign **consistent and unique IDs** to each player, regardless of which camera captured them. This helps in player-level tracking and analysis over time, across views.

---

## ✅ Objective

- Detect players in each frame using **YOLOv11**
- Extract **appearance-based embeddings** using ResNet18
- Compute **spatial and temporal association** for tracking
- Ensure consistent **player ID assignment** across camera views

---

## 📂 Folder Structure

Cross-Camera-ReID/
│
├── best.pt # Trained YOLOv11 weights
├── broadcast.mp4 # Camera 1 input video
├── tacticam.mp4 # Camera 2 input video
│
├── player_mapping.py # Main script for ID mapping
├── extract_features.py # Appearance + location features
├── enhanced_tracker.py # Tracker for re-identification
├── visualize.py # Draw bounding boxes + ID labels
├── evaluate.py # Performance evaluation script
│
├── outputs/
│ ├── broadcast_output.mp4
│ ├── tacticam_output.mp4
│ └── logs/ # Text logs with IDs per frame
│
└── README.md # You are here 🚀

---

## 🛠️ Setup Instructions

### 📦 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Ultralytics (for YOLOv11)
- scikit-learn

### 🔧 Installation


git clone https://github.com/<your-username>/Cross-Camera-ReID.git
cd Cross-Camera-ReID
pip install -r requirements.txt
If requirements.txt is missing, install manually:

pip install torch torchvision opencv-python numpy scikit-learn ultralytics `
🚀 Running the Project
▶️ Step 1: Run Player Detection and Tracking
bash
Copy
Edit
python player_mapping.py
This script does:

YOLOv11 detection on both videos

Feature extraction (bbox + ResNet embedding)

ID assignment using enhanced_tracker.py

Saves annotated videos to outputs/

Generates ID logs in outputs/logs/

📊 Step 2: Evaluate Performance
bash
Copy
Edit
python evaluate.py
You will get:

Re-ID Accuracy

Number of ID switches

Matching statistics

FPS (processing speed)

Results saved in evaluation_report.txt

🔍 Core Components
📌 YOLOv11 Detection (player_mapping.py)
Loads YOLOv11 weights (best.pt)

Detects bounding boxes for players

Filters based on confidence threshold

🧠 Feature Extraction (extract_features.py)
Uses bounding box and ResNet18 features

Captures appearance + spatial size for comparison

Feature = [x1, y1, area, ResNet_embedding]

🔁 Enhanced Tracker (enhanced_tracker.py)
Matches players across frames and videos

Uses cosine similarity between embeddings

Tracks based on appearance + motion smoothness

🖼️ Visualization (visualize.py)
Draws bounding boxes, IDs, and tracks on each frame

Color-codes players consistently

📈 Evaluation Metrics
Metric	Description
ReID Accuracy	How many players are assigned correct ID
ID Switches	When the same player is assigned a new ID
FPS	Frame Processing Speed

🧪 Example Output (Demo)


The same player (e.g., Player #3) retains the same ID across camera views.

🧠 Future Improvements
Integrate Deep SORT for more robust tracking

Use Siamese or Triplet networks for embedding

Introduce temporal smoothing (Kalman Filter)

Use camera calibration for precise spatial alignment

🤝 Acknowledgements
Ultralytics YOLOv11

ResNet18

Liat.ai for providing the assignment and challenge

📬 Contact
Pallavi Kottam
Artificial Intelligence & Data Science
📧 kottampallavi9@gmail.com
📍 India

🧠 If you found this project useful, give it a ⭐️ and share it!

---
