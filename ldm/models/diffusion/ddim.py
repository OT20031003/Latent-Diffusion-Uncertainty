"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    @torch.no_grad()
    def estimate_uncertainty(self, pred_x0, t, shape, num_samples=5, c=None, 
                             unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        Algorithm 1: Pixel-wise Uncertainty Estimation (Latent Space版)
        pred_x0: 現在のステップで予測された潜在表現 z0 [Batch, C, H, W]
        t: 現在のタイムステップ [Batch,]
        shape: 潜在表現の形状 (Batch, C, H, W)
        num_samples: モンテカルロサンプリング数 (M)。論文では M=5 推奨。
        c: 条件付け入力 (conditioning)
        """
        # アルファ値の取得 (Equation 2用)
        # tはバッチサイズ分のテンソルなので、代表値として最初の要素を取得
        t_idx = t[0].item()
        alpha_bar_t = self.model.alphas_cumprod[t_idx]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - alpha_bar_t)

        scores = []
        
        # M回ループして摂動を加えた潜在表現サンプルを作成し、スコアを計算
        for _ in range(num_samples):
            # 摂動 (Perturbation)
            # Equation 2: X_t^i = sqrt(alpha_bar) * z0 + sqrt(1-alpha_bar) * epsilon
            noise = torch.randn(shape, device=self.model.device)
            x_t_perturbed = sqrt_alpha_bar_t * pred_x0 + sqrt_one_minus_alpha_bar_t * noise
            
            # モデルに入力してスコア(epsilon)を計算
            # p_sample_ddimと同様にClassifier-Free Guidanceを考慮
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x_t_perturbed, t, c)
            else:
                x_in = torch.cat([x_t_perturbed] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                
            scores.append(e_t)

        # スタックして形状を (M, Batch, C, H, W) にする
        scores = torch.stack(scores)
        
        # 分散を計算 (Equation 8: Variance of the scores)
        # 次元0 (Mの方向) に沿って分散を取る。これが潜在空間上の不確実性マップとなる。
        uncertainty = torch.var(scores, dim=0)
        
        return uncertainty

    @torch.no_grad()
    def sample_awgn(self, S, batch_size, shape, noisy_latent, snr_db, callback=None,
                    normals_sequence=None, img_callback=None, quantize_x0=False,
                    eta=0., mask=None, x0=None, temperature=1., noise_dropout=0.,
                    score_corrector=None, corrector_kwargs=None, verbose=True,
                    log_every_t=100, unconditional_guidance_scale=1., unconditional_conditioning=None,
                    uncertainty_interval=10, # 不確実性を計算する間隔（ステップ数）
                    **kwargs):
        """
        AWGNチャネルの受信信号から復元を行う専用メソッド
        noisy_latent: 受信信号 y (z0 + noise)
        snr_db: チャネルのSNR (dB)
        戻り値: (samples, uncertainty_map)
        """
        
        # 1. 拡散モデルのSNRスケジュールを取得
        # alphas_cumprod は 1 -> 0 に向かって減っていく (signal power)
        alphas = self.model.alphas_cumprod.cpu().numpy()
        
        # 拡散過程における各時刻 t の SNR = alpha_t / (1 - alpha_t)
        diffusion_snrs = alphas / (1 - alphas)
        
        # 2. チャネルSNRに最も近い拡散時刻 t_start を探索
        target_snr = 10 ** (snr_db / 10.0)
        t_start_idx = np.abs(diffusion_snrs - target_snr).argmin()
        
        if verbose:
            print(f"Channel SNR: {snr_db} dB -> Starting Diffusion Step: {t_start_idx} (Model SNR: {diffusion_snrs[t_start_idx]:.2f})")

        # 3. DDIMスケジュールの作成 (Sステップに間引く)
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
        
        # 全1000ステップ中の t_start_idx に最も近い DDIMステップを探す
        closest_ddim_idx = np.abs(self.ddim_timesteps - t_start_idx).argmin()
        
        # DDIMサンプリング用のタイムステップリスト (逆順: T -> 0)
        timesteps = np.flip(self.ddim_timesteps)
        
        # 開始すべき時刻 t_start_ddim を探す
        start_idx = 0
        for i, t in enumerate(timesteps):
            if t <= t_start_idx:
                start_idx = i
                break
        
        # 4. 開始状態 x_T の準備
        # 受信信号 y を拡散モデルの状態 x_t にスケーリング
        alpha_at_start = alphas[t_start_idx]
        alpha_at_start = torch.tensor(alpha_at_start, device=self.model.device, dtype=torch.float32)
        
        # 開始点 x_start = sqrt(alpha) * y
        x_start = torch.sqrt(alpha_at_start) * noisy_latent
        
        # 5. サンプリングループ
        C, H, W = shape
        size = (batch_size, C, H, W)
        samples = x_start
        
        # 最終的な不確実性マップを保持する変数
        current_uncertainty = None

        # start_idx から最後までループ
        iterator = tqdm(timesteps[start_idx:], desc='AWGN Denoising', total=len(timesteps)-start_idx) if verbose else timesteps[start_idx:]

        for i, step in enumerate(iterator):
            index = len(timesteps) - start_idx - 1 - i
            ts = torch.full((batch_size,), step, device=self.model.device, dtype=torch.long)
            
            # 前のステップの画像を取得
            # p_sample_ddim は (x_prev, pred_x0) を返す仕様になっている前提
            outs = self.p_sample_ddim(samples, unconditional_conditioning, ts, index=index, use_original_steps=False,
                                      quantize_denoised=quantize_x0, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale)
            samples, pred_x0 = outs
            
            # === 不確実性の測定 (潜在空間) ===
            # 計算負荷軽減のため、指定した間隔または特定の条件下でのみ実行
            if i % uncertainty_interval == 0: 
                 current_uncertainty = self.estimate_uncertainty(
                     pred_x0, ts, size, 
                     num_samples=5, # 論文推奨値 M=5
                     c=unconditional_conditioning,
                     unconditional_guidance_scale=unconditional_guidance_scale,
                     unconditional_conditioning=unconditional_conditioning
                 )
            # ==============================
            
            if callback: callback(i)
            if img_callback: img_callback(samples, i)

        # 復元画像(潜在表現)と、最後に計算された不確実性マップを返す
        # 注意: 返り値が2つになったので、呼び出し元(img2img.pyなど)での受け取り方も変更が必要です
        return samples, current_uncertainty
    
    
   