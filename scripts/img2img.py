import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt

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
    """
    AWGNチャネルのシミュレーション
    """
    sig_power = torch.mean(latent ** 2, dim=(1, 2, 3), keepdim=True)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(latent) * noise_std
    noisy_latent = latent + noise
    return noisy_latent

def save_heatmap_with_correlation(uncertainty_tensor, error_tensor, output_dir, base_fname, target_size=(256, 256)):
    """
    不確実性マップをヒートマップ(Jet)として保存し、誤差との相関を計算する
    uncertainty_tensor: [Batch, C_lat, H_lat, W_lat] (潜在空間)
    error_tensor: [Batch, H_img, W_img] (画像空間でのチャンネル平均誤差)
    """
    if uncertainty_tensor is None:
        return

    # 1. 不確実性マップの前処理
    # チャンネル平均 -> [Batch, 1, H_lat, W_lat]
    unc_mean = torch.mean(uncertainty_tensor, dim=1, keepdim=True)
    
    # 画像サイズにアップサンプリング -> [Batch, 1, 256, 256]
    unc_upsampled = F.interpolate(unc_mean, size=target_size, mode='bilinear', align_corners=False)
    
    # [Batch, H, W] にスクイーズ
    unc_upsampled = unc_upsampled.squeeze(1)

    # CPUへ移動
    unc_np = unc_upsampled.cpu().numpy()
    err_np = error_tensor.cpu().numpy()

    correlations = []

    # バッチ内の各画像について処理
    for i in range(len(unc_np)):
        u_map = unc_np[i] # Uncertainty Map
        e_map = err_np[i] # Error Map (Channel Averaged)

        # --- 相関の計算 ---
        # フラット化
        u_flat = u_map.flatten()
        e_flat = e_map.flatten()
        
        # 相関係数 (Pearson)
        if np.std(u_flat) > 1e-6 and np.std(e_flat) > 1e-6:
            corr = np.corrcoef(u_flat, e_flat)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)

        # --- ヒートマップの保存 (Jet Colormap: Blue->Red) ---
        # 正規化 (0-1)
        u_min, u_max = u_map.min(), u_map.max()
        if u_max - u_min > 1e-6:
            u_norm = (u_map - u_min) / (u_max - u_min)
        else:
            u_norm = np.zeros_like(u_map)

        # カラーマップ適用 (matplotlibを使用)
        cmap = plt.get_cmap('jet')
        # cmap(u_norm) は [H, W, 4] (RGBA) を返す。0-1のfloat。
        colored_map = cmap(u_norm)
        
        # RGBA -> RGB, 0-255
        colored_img = (colored_map[:, :, :3] * 255).astype(np.uint8)
        
        img = Image.fromarray(colored_img)
        fname_no_ext = os.path.splitext(base_fname[i])[0]
        save_path = os.path.join(output_dir, f"{fname_no_ext}_uncertainty.png")
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

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in image_extensions])
    
    num_images = len(image_files)
    print(f"Found {num_images} images. Starting simulation with SNR={args.snr}dB...")

    all_correlations = []

    with torch.no_grad():
        for i in range(0, num_images, args.batch_size):
            batch_idx = i // args.batch_size
            
            # --- バッチデータの準備 ---
            batch_files = image_files[i : i + args.batch_size]
            current_batch_size = len(batch_files)
            
            batch_tensors = []
            for fname in batch_files:
                img_path = os.path.join(args.input_dir, fname)
                image = Image.open(img_path).convert("RGB")
                if image.size != (256, 256):
                    image = image.resize((256, 256), Image.BICUBIC)
                img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                batch_tensors.append(img_tensor)
            
            # 元画像 (Original Input): -1 ~ 1
            batch_input = torch.stack(batch_tensors).cuda()

            # --- 1. エンコード & チャネルシミュレーション ---
            encoder_posterior = model.encode_first_stage(batch_input)
            if isinstance(encoder_posterior, torch.Tensor):
                z0 = encoder_posterior
            elif hasattr(encoder_posterior, 'mode'):
                z0 = encoder_posterior.mode()
            else:
                z0 = encoder_posterior

            z_received = add_awgn_channel(z0, args.snr)

            # --- 2. 拡散モデルによる復元 ---
            samples, uncertainty_map = sampler.sample_awgn(
                S=args.ddim_steps,
                batch_size=current_batch_size, 
                shape=z0.shape[1:],
                noisy_latent=z_received,
                snr_db=args.snr,
                eta=0.0,
                verbose=True,
                uncertainty_interval=args.uncertainty_interval
            )

            # --- 3. デコード (復元画像) ---
            x_rec_batch = model.decode_first_stage(samples)
            # x_rec_batch はここで -1 ~ 1 の範囲 (clamp前)
            
            # --- 4. 実際の誤差 (Pixel-wise Error) の計算 ---
            # 比較のため両方を同じスケールで計算します (-1~1 のままで計算)
            # Squared Error: (Original - Reconstructed)^2
            diff = (batch_input - x_rec_batch) ** 2
            # チャンネル平均をとる -> [Batch, H, W]
            error_map_tensor = torch.mean(diff, dim=1)

            # --- 5. 保存と相関計算 ---
            # 画像保存用に 0~1 に正規化
            x_rec_norm = torch.clamp((x_rec_batch + 1.0) / 2.0, 0.0, 1.0)
            x_rec_np = x_rec_norm.cpu().permute(0, 2, 3, 1).numpy()
            
            for j, fname in enumerate(batch_files):
                x_rec_img = Image.fromarray((x_rec_np[j] * 255).astype(np.uint8))
                x_rec_img.save(os.path.join(args.output_dir, fname))

            # 不確実性マップの保存と相関計算
            # uncertainty_map (Latent) と error_map_tensor (Pixel, Channel Averaged) を渡す
            if uncertainty_map is not None:
                corrs = save_heatmap_with_correlation(
                    uncertainty_map, 
                    error_map_tensor, 
                    args.output_dir, 
                    batch_files
                )
                all_correlations.extend(corrs)
                print(f"Batch {batch_idx+1} Correlations: {['{:.4f}'.format(c) for c in corrs]}")

    if all_correlations:
        avg_corr = sum(all_correlations) / len(all_correlations)
        print(f"\nSimulation Finished.")
        print(f"Average Correlation (Uncertainty vs Error): {avg_corr:.4f}")
    else:
        print("\nSimulation Finished (No uncertainty map generated).")

if __name__ == "__main__":
    main()