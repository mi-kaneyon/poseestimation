import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 事前訓練済みの深度推定モデルのロード
model = fcn_resnet50(pretrained=True).eval().to(device)

# 画像前処理のためのトランスフォーム
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def estimate_depth(frame):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']
    
    # 深度マップの取得
    depth_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    return depth_map

def generate_point_cloud(depth_map, scale=0.05):
    h, w = depth_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map * scale

    # 点群の生成
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).transpose()
    
    return points

def visualize_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis')
    plt.show()

def main():
    cap = cv2.VideoCapture(0)  # デバイスIDを0に設定

    if not cap.isOpened():
        print("カメラを開くことができません。デバイスIDを確認してください。")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("フレームを取得できません。")
            break
        
        # 深度推定
        depth_map = estimate_depth(frame)
        
        # 点群生成
        points = generate_point_cloud(depth_map)
        
        # 点群の可視化
        visualize_point_cloud(points)
        
        # フレームを表示
        cv2.imshow('Frame', frame)
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
