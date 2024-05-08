import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv

# Device selection for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MediaPipe solutions with enhanced settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.75,  # 顔の検出信頼度を高く設定
    min_tracking_confidence=0.75    # 顔の追跡信頼度も同様に高く設定
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# PyTorch transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define body parts indices and their corresponding TMU weights
body_parts = {
    'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'left_arm': [11, 13, 15, 17, 19, 21],
    'right_arm': [12, 14, 16, 18, 20, 22],
    'body_core': [11, 12, 23, 24],
    'legs': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
}
weights = {
    'face': 0.8,
    'left_arm': 1.0,
    'right_arm': 1.0,
    'body_core': 1.2,
    'legs': 1.4
}

# Open CSV file for logging
with open('movement_log.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Part', 'X', 'Y', 'Visibility', 'Timestamp'])

    # Video capture with specific settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert frame to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()

        # Process with MediaPipe solutions
        with torch.no_grad():
            pose_results = pose.process(frame_rgb)
            face_results = face.process(frame_rgb)
            hands_results = hands.process(frame_rgb)

        # Log pose landmarks with body part classification
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                body_part = next((part for part, indices in body_parts.items() if idx in indices), 'unknown')
                part_name = f"{body_part}_Landmark_{idx}"
                csv_writer.writerow([part_name, landmark.x, landmark.y, landmark.visibility, timestamp])

        # Convert to PyTorch tensor and send to GPU
        frame_tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)

        # Display the annotated frame
        cv2.imshow('Pose, Face, and Hands Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
