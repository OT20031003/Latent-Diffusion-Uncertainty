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

def create_semantic_uncertainty_mask(uncertainty_map, parsing_map, retransmission_rate=0.1):
    """
    Latent空間上でセグメントごとの不確実性を評価し、正確なレートでマスクを作成する関数。
    
    Args:
        uncertainty_map (np.ndarray): Latent空間の不確実性マップ (H_lat, W_lat) または (C, H_lat, W_lat)
        parsing_map (np.ndarray): ピクセル空間のセグメンテーションマップ (H_img, W_img)
        retransmission_rate (float): 再送率 (0.0 - 1.0)
    
    Returns:
        np.ndarray: Latent空間サイズのマスク (H_lat, W_lat), 値は 0.0 または 1.0
    """
    # 1. Latent空間のサイズを取得
    if uncertainty_map.ndim == 3:
        # (C, H, W) の場合、チャンネル方向を平均して (H, W) にする
        u_map_latent = np.mean(uncertainty_map, axis=0)
    else:
        u_map_latent = uncertainty_map
        
    h_lat, w_lat = u_map_latent.shape

    # 2. セグメンテーションマップをLatentサイズにダウンサンプリング
    # クラスID(整数)を壊さないよう、必ず INTER_NEAREST を使用する
    parsing_map_latent = cv2.resize(parsing_map, (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)
    
    # 3. セグメントごとのスコア計算 (全てLatent空間で計算)
    segment_scores = {}
    present_labels = np.unique(parsing_map_latent)
    
    for label in present_labels:
        # 背景(0)などを除外したい場合はここで制御可能
        # if label == 0: continue
        
        mask_segment = (parsing_map_latent == label)
        if np.sum(mask_segment) > 0:
            # そのセグメントに属するLatentピクセルの不確実性平均をスコアとする
            segment_scores[label] = u_map_latent[mask_segment].mean()
            
    # 不確実性が高い順にセグメントをソート
    sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 4. 予算に合わせてマスクを作成
    final_mask = np.zeros_like(u_map_latent, dtype=np.float32)
    
    total_pixels = h_lat * w_lat
    max_allowed_pixels = int(total_pixels * retransmission_rate)
    current_pixel_count = 0
    
    for label, score in sorted_segments:
        if current_pixel_count >= max_allowed_pixels:
            break
            
        mask_segment = (parsing_map_latent == label)
        segment_pixels = np.sum(mask_segment)
        
        # セグメント丸ごと追加OKなら追加
        if current_pixel_count + segment_pixels <= max_allowed_pixels:
            final_mask[mask_segment] = 1.0
            current_pixel_count += segment_pixels
            
        # 予算オーバーする場合は、セグメント内の画素を不確実性順に選抜
        else:
            remaining = max_allowed_pixels - current_pixel_count
            if remaining > 0:
                # このセグメント内の座標を取得
                y_idxs, x_idxs = np.where(mask_segment)
                
                # その場所の不確実性値を取得
                vals = u_map_latent[mask_segment]
                
                # セグメント内で不確実性が高い順にソートして上位のみ選択
                local_sorted_indices = np.argsort(vals)[::-1] # 降順
                selected_local_indices = local_sorted_indices[:remaining]
                
                # 選択されたピクセルをマスクに追加
                final_mask[y_idxs[selected_local_indices], x_idxs[selected_local_indices]] = 1.0
            
            # 予算を満たしたので終了
            break
            
    return final_mask
def create_semantic_weighted_mask(uncertainty_map, parsing_map, retransmission_rate=0.1):
    """
    パーツごとの重要度に基づき不確実性を重み付けし、重要な領域（目、鼻、口）を優先してマスクを作成する。
    
    Args:
        uncertainty_map (np.ndarray): Latent空間の不確実性マップ (C, H_lat, W_lat) または (H_lat, W_lat)
        parsing_map (np.ndarray): ピクセル空間のセグメンテーションマップ (H_img, W_img)
        retransmission_rate (float): 再送率 (0.0 - 1.0)
    """
    # 1. Latent空間のサイズを取得 & マップの準備
    if uncertainty_map.ndim == 3:
        u_map_latent = np.mean(uncertainty_map, axis=0)
    else:
        u_map_latent = uncertainty_map
        
    h_lat, w_lat = u_map_latent.shape

    # 2. セグメンテーションマップをLatentサイズにダウンサンプリング (Nearest Neighbor)
    parsing_map_latent = cv2.resize(parsing_map, (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)

    # 3. 重要度ウェイトの定義 (BiSeNetのクラスID準拠)
    # 値が大きいほど優先的に再送される
    # 0:bg, 1:skin, 2-3:brows, 4-5:eyes, 10:nose, 11-13:mouth, 17:hair
    weights = {
        # === 最優先 (構造とIDの核) ===
        4: 6.0, 5: 6.0,   # 目
        2: 5.0, 3: 5.0,   # 眉
        10: 5.0,          # 鼻
        11: 5.0, 12: 5.0, 13: 5.0, # 口・唇

        # === 優先 (顔の形状) ===
        1: 2.0,           # 肌 (幻覚が出やすい頬など)
        
        # === 低優先 (テクスチャ) ===
        17: 0.8,          # 髪 (高周波ノイズでUncertaintyが高くなりがちなので下げる)
        16: 0.5,          # 服
        
        # === 除外対象 ===
        0: 0.1            # 背景
    }
    
    # デフォルトの重み
    default_weight = 0.5 

    # 4. 重みマップの作成
    weight_map = np.full_like(u_map_latent, default_weight)
    
    for class_id, w in weights.items():
        weight_map[parsing_map_latent == class_id] = w
        
    # 5. スコアの計算 (不確実性 × 重要度)
    # これにより、不確実性が低くても目が重要ならスコアが高くなる
    weighted_score = u_map_latent * weight_map
    
    # 6. 上位R%を選抜してマスク生成
    flat_scores = weighted_score.flatten()
    k = int(flat_scores.size * retransmission_rate)
    
    final_mask = np.zeros_like(u_map_latent, dtype=np.float32)
    
    if k > 0:
        # 上位k個の閾値を求めてマスク化
        # (np.partitionを使うと高速にk番目の値を見つけられます)
        threshold_idx = flat_scores.size - k
        threshold = np.partition(flat_scores, threshold_idx)[threshold_idx]
        
        final_mask = (weighted_score >= threshold).astype(np.float32)
        
    return final_mask