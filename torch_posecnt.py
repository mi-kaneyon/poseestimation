import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv
import time

def main():
    # GPUの有無でデバイスを選択
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MediaPipeの各ソリューションの初期化
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # コンテキストマネージャを利用して各ソリューションのライフサイクルを管理
    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.75, min_tracking_confidence=0.75) as face, \
         mp_hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

        # PyTorch用の画像前処理（GPU処理のためのテンソル変換）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Body parts の定義（必要に応じてインデックスは調整してください）
        body_parts = {
            'face': list(range(0, 11)),
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'body_core': [11, 12, 23, 24],
            'legs': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        }
        # TMUに対する重み（例）
        weights = {
            'face': 0.8,
            'left_arm': 1.0,
            'right_arm': 1.0,
            'body_core': 1.2,
            'legs': 1.4
        }

        # CSVファイル名を実行時の日時で動的に設定
        csv_filename = f"movement_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Part', 'X', 'Y', 'Visibility', 'Timestamp'])

            # カメラキャプチャの設定
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error opening video stream or file")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            print("カメラを起動しました。終了する場合は 'q' キーを押してください。")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("フレームが取得できません。終了します。")
                        break

                    # BGR画像をRGBに変換
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 現在時刻（秒）をタイムスタンプとして取得
                    timestamp = time.time()

                    # PyTorchの勾配計算を無効にして高速化
                    with torch.no_grad():
                        pose_results = pose.process(frame_rgb)
                        face_results = face.process(frame_rgb)
                        hands_results = hands.process(frame_rgb)

                    # Poseのランドマークが検出された場合、描画とCSVへの書き込みを実施
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            body_part = next((part for part, indices in body_parts.items() if idx in indices), 'unknown')
                            part_name = f"{body_part}_Landmark_{idx}"
                            csv_writer.writerow([part_name, landmark.x, landmark.y, landmark.visibility, timestamp])

                    # 例としてフレームをPyTorch用テンソルに変換（GPU処理に利用可能）
                    frame_tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)

                    cv2.imshow('Pose, Face, and Hands Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            except KeyboardInterrupt:
                print("ユーザーによって中断されました。")
            finally:
                cap.release()
                cv2.destroyAllWindows()
                print(f"ログは {csv_filename} に保存されました。")

if __name__ == "__main__":
    main()
