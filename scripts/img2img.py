import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

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
    AWGNチャネルのシミュレーション (バッチ対応版)
    latent: 送信信号 (z0) [Batch, C, H, W]
    snr_db: 目標SNR (dB)
    """
    # 信号電力の計算
    # バッチ処理の場合、画像ごとに電力が異なる可能性があるため、
    # dim=(1, 2, 3) で各サンプルの平均電力を計算し、keepdim=True で形状を [B, 1, 1, 1] に保つ
    sig_power = torch.mean(latent ** 2, dim=(1, 2, 3), keepdim=True)
    
    # SNR(dB) = 10 * log10(P_signal / P_noise)
    # => P_noise = P_signal / 10^(SNR/10)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise_std = torch.sqrt(noise_power)
    
    # ノイズ生成 (latentと同じ形状の正規分布ノイズ)
    noise = torch.randn_like(latent) * noise_std
    
    # 受信信号 y = x + n
    noisy_latent = latent + noise
    return noisy_latent

def main():
    parser = argparse.ArgumentParser(description="DiffCom AWGN Simulation img2img")
    parser.add_argument("--input_dir", type=str, default="input_dir", help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, default="results/", help="Path to output directory")
    parser.add_argument("--snr", type=float, default=-10.0, help="Channel SNR in dB")
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", help="Path to model config")
    parser.add_argument("--ckpt", type=str, default="models/ldm/ffhq256/model.ckpt", help="Path to model checkpoint")
    parser.add_argument("--ddim_steps", type=int, default=200, help="Number of DDIM sampling steps")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for processing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # モデルのロード
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    sampler = DDIMSampler(model)

    # 入力画像の取得
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in image_extensions])
    
    num_images = len(image_files)
    print(f"Found {num_images} images. Starting simulation with SNR={args.snr}dB, Batch Size={args.batch_size}...")

    # バッチ数の計算
    total_batches = (num_images + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        # tqdmを使わず、通常のrangeループを使用
        for i in range(0, num_images, args.batch_size):
            batch_idx = i // args.batch_size
            
            # バッチ進捗のログ出力
            print(f"Processing Batch {batch_idx + 1}/{total_batches} (Images {i} - {min(i + args.batch_size, num_images)})")

            # 現在のバッチに含まれるファイルリストを取得
            batch_files = image_files[i : i + args.batch_size]
            current_batch_size = len(batch_files)
            
            # バッチ内の画像を読み込んでリストに格納
            batch_tensors = []
            for fname in batch_files:
                img_path = os.path.join(args.input_dir, fname)
                
                # 画像読み込み & 前処理 (-1 ~ 1)
                image = Image.open(img_path).convert("RGB")
                w, h = image.size
                if w != 256 or h != 256:
                    image = image.resize((256, 256), Image.BICUBIC)
                
                img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
                # [H, W, C] -> [C, H, W]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                batch_tensors.append(img_tensor)
            
            # リストを結合して [Batch, C, H, W] のテンソルを作成
            batch_input = torch.stack(batch_tensors).cuda()

            # 1. エンコード (Source Coding)
            encoder_posterior = model.encode_first_stage(batch_input)
            
            # Tensorかどうかの判定 (VQ-GAN対応)
            if isinstance(encoder_posterior, torch.Tensor):
                z0 = encoder_posterior
            elif hasattr(encoder_posterior, 'mode'):
                z0 = encoder_posterior.mode()
            else:
                z0 = encoder_posterior

            # 2. AWGNチャネル (Channel Simulation)
            z_received = add_awgn_channel(z0, args.snr)

            # 3. 拡散モデルによる復元 (Decoder / Denoiser)
            # verbose=True にして、DDIMサンプリングのプログレスバーを表示させる
            samples, _ = sampler.sample_awgn(
                S=args.ddim_steps,
                batch_size=current_batch_size, 
                shape=z0.shape[1:],
                noisy_latent=z_received,
                snr_db=args.snr,
                eta=0.0,
                verbose=True 
            )

            # 4. 画像空間へデコード
            x_rec_batch = model.decode_first_stage(samples)
            x_rec_batch = torch.clamp((x_rec_batch + 1.0) / 2.0, 0.0, 1.0) # 0~1に正規化

            # 5. 保存 (バッチ内の各画像を個別に保存)
            x_rec_batch = x_rec_batch.cpu().permute(0, 2, 3, 1).numpy()
            
            for j, fname in enumerate(batch_files):
                x_rec_img = Image.fromarray((x_rec_batch[j] * 255).astype(np.uint8))
                x_rec_img.save(os.path.join(args.output_dir, fname))

    print("Simulation finished.")

if __name__ == "__main__":
    main()