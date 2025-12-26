import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import sys
import os

# face-parsing.PyTorchへのパスを通す (環境に合わせて調整してください)
sys.path.append(os.path.join(os.path.dirname(__file__), '../face-parsing.PyTorch'))

from model import BiSeNet

class FaceParser:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.n_classes = 19
        self.net = BiSeNet(n_classes=self.n_classes)
        
        # 学習済みモデルのロード
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.to(device)
        self.net.eval()

        # Face Parsing用の正規化
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_parsing_map(self, img_tensor):
        """
        img_tensor: VAEからデコードされた画像 (B, C, H, W) 範囲 [-1, 1]
        戻り値: (H, W) のnumpy配列 (各ピクセルに0-18のID)
        """
        # 1. [-1, 1] -> [0, 255] uint8 に変換
        img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img + 1.0) / 2.0 * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 2. モデル入力サイズ(512x512)にリサイズ
        original_h, original_w = img.shape[:2]
        img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # 3. 推論
        inp = self.to_tensor(img_resized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(inp)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
        # 4. 元の画像サイズに戻す (最近傍補間)
        parsing_original = cv2.resize(parsing.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        return parsing_original

def create_semantic_uncertainty_mask(uncertainty_map, parsing_map, retransmission_rate=0.2):
    """
    セグメントごとの不確実性を計算し、指定されたレートに収まるようにマスクを作成する関数。
    重要度ウェイトは使用せず、純粋な不確実性平均スコアで判断します。
    """
    # uncertainty_map (Latent) を parsing_map (Pixel) のサイズに拡大
    h, w = parsing_map.shape
    u_map_resized = cv2.resize(uncertainty_map, (w, h), interpolation=cv2.INTER_NEAREST)
    
    segment_scores = {}
    present_labels = np.unique(parsing_map)
    
    # 各パーツの不確実性平均を計算
    for label in present_labels:
        # 背景(0)を除外したい場合はここで continue
        # if label == 0: continue 
        
        mask = (parsing_map == label)
        if np.sum(mask) > 0:
            segment_scores[label] = u_map_resized[mask].mean()
            
    # 不確実性が高い順にセグメントをソート
    sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_mask = np.zeros_like(parsing_map, dtype=np.float32)
    
    # 許容される最大ピクセル数
    total_pixels = h * w
    max_allowed_pixels = int(total_pixels * retransmission_rate)
    current_pixel_count = 0
    
    for label, score in sorted_segments:
        # 既に予算がいっぱいの場合は終了
        if current_pixel_count >= max_allowed_pixels:
            break
            
        mask = (parsing_map == label)
        segment_pixels = np.sum(mask)
        
        # このセグメントを丸ごと追加しても予算内に収まる場合
        if current_pixel_count + segment_pixels <= max_allowed_pixels:
            final_mask[mask] = 1.0
            current_pixel_count += segment_pixels
            
        # このセグメントを追加すると予算オーバーする場合 -> 間引き処理
        else:
            remaining_budget = max_allowed_pixels - current_pixel_count
            if remaining_budget > 0:
                # このセグメント内のピクセル座標を取得
                y_indices, x_indices = np.where(mask)
                
                # このセグメント内の不確実性値を取得
                segment_uncertainties = u_map_resized[mask]
                
                # セグメント内で不確実性が高い順にインデックスを取得 (argsortは昇順なので反転)
                sorted_indices_local = np.argsort(segment_uncertainties)[::-1]
                
                # 予算分だけ上位のピクセルを選択
                selected_indices_local = sorted_indices_local[:remaining_budget]
                
                # グローバル座標にマッピングしてマスクをオンにする
                selected_y = y_indices[selected_indices_local]
                selected_x = x_indices[selected_indices_local]
                final_mask[selected_y, selected_x] = 1.0
                
            # 予算を満たしたので終了
            break
            
    return final_mask