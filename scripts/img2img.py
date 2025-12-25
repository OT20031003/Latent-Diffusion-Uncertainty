import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt

# --- 外部ライブラリのインポート (なければスキップ) ---
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: 'lpips' not found. LPIPS metric will be skipped.")

try:
    from DISTS_pytorch import DISTS
    DISTS_AVAILABLE = True
except ImportError:
    DISTS_AVAILABLE = False
    print("Warning: 'DISTS_pytorch' not found. DISTS metric will be skipped.")

try:
    from facenet_pytorch import InceptionResnetV1
    IDLOSS_AVAILABLE = True
except ImportError:
    IDLOSS_AVAILABLE = False
    print("Warning: 'facenet-pytorch' not found. ID Loss will be skipped.")
    print("To install: pip install facenet-pytorch")


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}...")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        print("missing keys:", u)
        print("unexpected keys:", m)
    model.cuda()
    model.eval()
    return model

def add_awgn_channel(latent, snr_db):
    """AWGNチャネルのシミュレーション"""
    sig_power = torch.mean(latent ** 2, dim=(1, 2, 3), keepdim=True)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(latent) * noise_std
    noisy_latent = latent + noise
    return noisy_latent

def calculate_psnr(img1, img2):
    """PSNR計算 (入力: [0, 1])"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_id_loss(img1, img2, net):
    """ID Loss計算 (Cosine Similarityの逆数など)"""
    # img: [B, C, H, W], range [0, 1] -> facenetは [-1, 1] などを期待する場合が多いが
    # facenet-pytorchの実装では標準化が必要な場合がある。ここでは簡易的に 0-1 -> -1~1 にして入力
    # またサイズは160x160推奨だが、可変でも動く
    
    # Preprocessing
    img1_in = (img1 * 2) - 1.0
    img2_in = (img2 * 2) - 1.0
    
    # Resize to 160x160 if needed (optional but recommended for FaceNet)
    img1_in = F.interpolate(img1_in, size=(160, 160), mode='bilinear', align_corners=False)
    img2_in = F.interpolate(img2_in, size=(160, 160), mode='bilinear', align_corners=False)

    with torch.no_grad():
        emb1 = net(img1_in) # [B, 512]
        emb2 = net(img2_in) # [B, 512]
    
    # Cosine Similarity
    cosine_sim = F.cosine_similarity(emb1, emb2)
    # ID Loss = 1 - Cosine Similarity
    id_loss = 1.0 - cosine_sim
    
    return id_loss.item()


def save_heatmap_with_correlation(uncertainty_tensor, error_tensor, output_dir, base_fname, step_label, target_size=(256, 256)):
    """
    不確実性マップを保存し、誤差との相関を計算して返す
    Returns: List of correlation coefficients (one per image in batch)
    """
    if uncertainty_tensor is None:
        return [0.0] * len(error_tensor)

    # 前処理
    unc_mean = torch.mean(uncertainty_tensor, dim=1, keepdim=True)
    unc_upsampled = F.interpolate(unc_mean, size=target_size, mode='bilinear', align_corners=False)
    unc_upsampled = unc_upsampled.squeeze(1)

    unc_np = unc_upsampled.cpu().numpy()
    err_np = error_tensor.cpu().numpy()

    correlations = []

    for i in range(len(unc_np)):
        u_map = unc_np[i]
        e_map = err_np[i]

        # 相関計算
        u_flat = u_map.flatten()
        e_flat = e_map.flatten()
        
        if np.std(u_flat) > 1e-6 and np.std(e_flat) > 1e-6:
            corr = np.corrcoef(u_flat, e_flat)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)

        # ヒートマップ保存
        u_min, u_max = u_map.min(), u_map.max()
        if u_max - u_min > 1e-6:
            u_norm = (u_map - u_min) / (u_max - u_min)
        else:
            u_norm = np.zeros_like(u_map)

        cmap = plt.get_cmap('jet')
        colored_map = cmap(u_norm)
        colored_img = (colored_map[:, :, :3] * 255).astype(np.uint8)
        
        img = Image.fromarray(colored_img)
        fname_no_ext = os.path.splitext(base_fname[i])[0]

        specific_dir = os.path.join(output_dir, str(i))
        os.makedirs(specific_dir, exist_ok=True)

        save_path = os.path.join(specific_dir, f"{fname_no_ext}_unc_{step_label}.png")
        img.save(save_path)

    return correlations

def main():
    parser = argparse.ArgumentParser(description="DiffCom AWGN Simulation img2img")
    parser.add_argument("--input_dir", type=str, default="input_dir", help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, default="results/", help="Path to output directory")
    parser.add_argument("--snr", type=float, default=-10.0, help="Channel SNR in dB")
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", help="Path to model config")
    parser.add_argument("--ckpt", type=str, default="models/ldm/ffhq256/model.ckpt", help="Path to model checkpoint")
    parser.add_argument("--ddim_steps", type=int, default=200, help="Number of DDIM sampling steps")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for processing")
    parser.add_argument("--uncertainty_interval", type=int, default=10, help="Interval steps to calculate uncertainty")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    sampler = DDIMSampler(model)

    # --- 評価モデルの準備 ---
    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='alex').cuda().eval()
    
    loss_fn_dists = None
    if DISTS_AVAILABLE:
        loss_fn_dists = DISTS().cuda().eval()

    loss_fn_id = None
    if IDLOSS_AVAILABLE:
        print("Loading FaceNet model for ID Loss...")
        # vggface2 で事前学習されたモデルを使用
        loss_fn_id = InceptionResnetV1(pretrained='vggface2').cuda().eval()

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in image_extensions])
    
    num_images = len(image_files)
    print(f"Found {num_images} images. Starting simulation with SNR={args.snr}dB...")

    with torch.no_grad():
        for i in range(0, num_images, args.batch_size):
            batch_idx = i // args.batch_size
            batch_out_dir = os.path.join(args.output_dir, f"snr{args.snr}dB", f"batch{batch_idx}")
            os.makedirs(batch_out_dir, exist_ok=True)
            
            batch_files = image_files[i : i + args.batch_size]
            current_batch_size = len(batch_files)
            
            # 画像読み込み
            batch_tensors = []
            for fname in batch_files:
                img_path = os.path.join(args.input_dir, fname)
                image = Image.open(img_path).convert("RGB")
                if image.size != (256, 256):
                    image = image.resize((256, 256), Image.BICUBIC)
                img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                batch_tensors.append(img_tensor)
            
            batch_input = torch.stack(batch_tensors).cuda() # [-1, 1]

            # 1. Encode & Channel
            encoder_posterior = model.encode_first_stage(batch_input)
            if isinstance(encoder_posterior, torch.Tensor):
                z0 = encoder_posterior
            elif hasattr(encoder_posterior, 'mode'):
                z0 = encoder_posterior.mode()
            else:
                z0 = encoder_posterior

            z_received = add_awgn_channel(z0, args.snr)

            # 2. Diffusion Sampling
            samples, uncertainty_history = sampler.sample_awgn(
                S=args.ddim_steps,
                batch_size=current_batch_size, 
                shape=z0.shape[1:],
                noisy_latent=z_received,
                snr_db=args.snr,
                eta=0.0,
                verbose=True,
                uncertainty_interval=args.uncertainty_interval
            )

            # 3. Decode
            x_rec_batch = model.decode_first_stage(samples)
            
            # --- 4. 評価指標の計算と保存 ---
            # 比較用に [0, 1] に正規化
            gt_01 = torch.clamp((batch_input + 1.0) / 2.0, 0.0, 1.0)
            rec_01 = torch.clamp((x_rec_batch + 1.0) / 2.0, 0.0, 1.0)

            # 誤差マップ (相関計算用)
            diff = (batch_input - x_rec_batch) ** 2
            error_map_tensor = torch.mean(diff, dim=1) # [B, H, W]

            # 画像ごとの結果格納用辞書
            batch_results = {}
            for fname in batch_files:
                batch_results[fname] = {
                    "metrics": {},
                    "correlations": {} # step -> correlation
                }

            # 画像ごとのメトリクス計算
            for j, fname in enumerate(batch_files):
                # Save Image
                x_rec_np = rec_01[j].cpu().permute(1, 2, 0).numpy()
                x_rec_img = Image.fromarray((x_rec_np * 255).astype(np.uint8))
                
                specific_out_dir = os.path.join(batch_out_dir, str(j))
                os.makedirs(specific_out_dir, exist_ok=True)
                x_rec_img.save(os.path.join(specific_out_dir, fname))

                # Metrics
                # Input tensors: [1, C, H, W]
                img_gt = gt_01[j:j+1]
                img_rec = rec_01[j:j+1]

                # PSNR
                psnr = calculate_psnr(img_gt, img_rec)
                batch_results[fname]["metrics"]["psnr"] = psnr.item() if isinstance(psnr, torch.Tensor) else psnr

                # LPIPS
                if loss_fn_lpips is not None:
                    # LPIPS expects [-1, 1]
                    lpips_val = loss_fn_lpips(img_gt * 2 - 1, img_rec * 2 - 1)
                    batch_results[fname]["metrics"]["lpips"] = lpips_val.item()
                
                # DISTS
                if loss_fn_dists is not None:
                    dists_val = loss_fn_dists(img_gt, img_rec)
                    batch_results[fname]["metrics"]["dists"] = dists_val.item()

                # ID Loss
                if loss_fn_id is not None:
                    id_loss = calculate_id_loss(img_gt, img_rec, loss_fn_id)
                    batch_results[fname]["metrics"]["id_loss"] = id_loss

            # --- 5. 不確実性マップの相関計算と保存 ---
            print(f"--- Batch {batch_idx+1} Correlations ---")
            if uncertainty_history:
                for step_idx, unc_map in uncertainty_history:
                    step_label = f"step{step_idx:04d}"
                    
                    # 各画像の相関係数リストを取得
                    corrs = save_heatmap_with_correlation(
                        unc_map, 
                        error_map_tensor, 
                        batch_out_dir, 
                        batch_files, 
                        step_label
                    )
                    
                    # JSON用に格納
                    for j, fname in enumerate(batch_files):
                        batch_results[fname]["correlations"][step_label] = corrs[j]
                    
                    avg_corr = sum(corrs) / len(corrs) if corrs else 0
                    print(f"Step {step_idx}: Avg Correlation = {avg_corr:.4f}")

            # --- 6. JSON保存 ---
            json_path = os.path.join(batch_out_dir, "metrics.json")
            
            # numpy型などをjson serializableに変換
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating, np.bool_)):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)

            with open(json_path, 'w') as f:
                json.dump(batch_results, f, indent=4, cls=NumpyEncoder)
            print(f"Metrics saved to {json_path}")

    print("\nSimulation Finished.")

if __name__ == "__main__":
    main()