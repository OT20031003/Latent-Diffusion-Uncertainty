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

def save_heatmap_with_correlation(uncertainty_tensor, error_tensor, output_dir, base_fname, step_label, target_size=(256, 256)):
    """
    不確実性マップをヒートマップ(Jet)として保存し、誤差との相関を計算する
    output_dir: バッチのルートディレクトリ (例: results/snr-10.0dB/batch0/)
    """
    if uncertainty_tensor is None:
        return []

    # 1. 不確実性マップの前処理
    unc_mean = torch.mean(uncertainty_tensor, dim=1, keepdim=True)
    unc_upsampled = F.interpolate(unc_mean, size=target_size, mode='bilinear', align_corners=False)
    unc_upsampled = unc_upsampled.squeeze(1)

    # CPUへ移動
    unc_np = unc_upsampled.cpu().numpy()
    err_np = error_tensor.cpu().numpy()

    correlations = []

    # バッチ内の各画像について処理
    for i in range(len(unc_np)):
        u_map = unc_np[i]
        e_map = err_np[i]

        # --- 相関の計算 ---
        u_flat = u_map.flatten()
        e_flat = e_map.flatten()
        
        if np.std(u_flat) > 1e-6 and np.std(e_flat) > 1e-6:
            corr = np.corrcoef(u_flat, e_flat)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)

        # --- ヒートマップの保存 ---
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

        # 保存先ディレクトリの作成: output_dir/{index}/
        # main関数で既に作成されているはずだが、念のため ensure する
        specific_dir = os.path.join(output_dir, str(i))
        os.makedirs(specific_dir, exist_ok=True)

        # ファイル名にステップ数を含める
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

    # args.output_dir はベースディレクトリとして使用 (例: results/)
    os.makedirs(args.output_dir, exist_ok=True)

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    sampler = DDIMSampler(model)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in image_extensions])
    
    num_images = len(image_files)
    print(f"Found {num_images} images. Starting simulation with SNR={args.snr}dB...")

    with torch.no_grad():
        for i in range(0, num_images, args.batch_size):
            batch_idx = i // args.batch_size
            
            # バッチごとの出力ディレクトリ: results/snr{SNR}dB/batch{ID}/
            batch_out_dir = os.path.join(args.output_dir, f"snr{args.snr}dB", f"batch{batch_idx}")
            
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
            
            batch_input = torch.stack(batch_tensors).cuda()

            # 1. エンコード & チャネル
            encoder_posterior = model.encode_first_stage(batch_input)
            if isinstance(encoder_posterior, torch.Tensor):
                z0 = encoder_posterior
            elif hasattr(encoder_posterior, 'mode'):
                z0 = encoder_posterior.mode()
            else:
                z0 = encoder_posterior

            z_received = add_awgn_channel(z0, args.snr)

            # 2. 拡散モデル (履歴を受け取る)
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

            # 3. デコード
            x_rec_batch = model.decode_first_stage(samples)
            
            # 4. 誤差計算
            diff = (batch_input - x_rec_batch) ** 2
            error_map_tensor = torch.mean(diff, dim=1) # [B, H, W]

            # 5. 画像保存 (ディレクトリ構造対応)
            x_rec_norm = torch.clamp((x_rec_batch + 1.0) / 2.0, 0.0, 1.0)
            x_rec_np = x_rec_norm.cpu().permute(0, 2, 3, 1).numpy()
            
            for j, fname in enumerate(batch_files):
                x_rec_img = Image.fromarray((x_rec_np[j] * 255).astype(np.uint8))
                
                # 画像ごとのディレクトリ: results/snr-10.0dB/batch0/{index}/
                specific_out_dir = os.path.join(batch_out_dir, str(j))
                os.makedirs(specific_out_dir, exist_ok=True)
                
                x_rec_img.save(os.path.join(specific_out_dir, fname))

            # 6. 全ステップの不確実性について相関計算 & 保存
            print(f"\n--- Batch {batch_idx+1} Analysis ---")
            if uncertainty_history:
                # ステップごとにループ
                for step_idx, unc_map in uncertainty_history:
                    step_label = f"step{step_idx:04d}"
                    
                    # ディレクトリパスは batch_out_dir を渡し、関数内で index を付与する
                    corrs = save_heatmap_with_correlation(
                        unc_map, 
                        error_map_tensor, 
                        batch_out_dir, 
                        batch_files,
                        step_label
                    )
                    
                    # ログ出力
                    avg_corr = sum(corrs) / len(corrs) if corrs else 0
                    print(f"Step {step_idx}: Avg Correlation = {avg_corr:.4f}")
            else:
                print("No uncertainty history found.")

    print("\nSimulation Finished.")

if __name__ == "__main__":
    main()