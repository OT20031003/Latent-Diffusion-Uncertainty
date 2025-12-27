# DiffCom Retransmission Simulation (img2img.py)

このリポジトリは、Latent Diffusion Model (LDM) を用いた「意味的通信（Semantic Communication）」および「再送制御（Retransmission）」のシミュレーション環境です。

通信路上のノイズ（AWGN）によって劣化した画像を、**受信側で推定した「不確実性（Uncertainty）」**や**「セマンティック情報（顔のパーツ）」**に基づいて部分的に再送要求し、インペインティング（Inpainting）技術を用いて修復します。

---

## 🛠 環境構築 (Installation)

### 1. Python環境のセットアップ
提供されている `environment.yaml` を使用して conda 環境を作成します。

```
conda env create -f environment.yaml
conda activate ldm
```
### 2. 必須ライブラリの追加インストール評価指標の計算や顔解析のために、以下のライブラリを追加でインストールしてください。Bash# LPIPS, DISTS, ID Loss (FaceNet), FIDなどの評価用ライブラリ
```
pip install lpips DISTS-pytorch facenet-pytorch pytorch-fid
```
### 3. Face Parsing (Semantic Segmentation) のセットアップsemantic 系の手法を使用する場合、顔のパーツ（目、口、鼻など）を特定するために face-parsing.PyTorch が必要

プロジェクトのルートディレクトリに face-parsing.PyTorch をクローンします。学習済みモデル（79999_iter.pth）をダウンロードして models/face_parsing/ 等に配置します。Bash# リポジトリのクローン
```
git clone https://github.com/zllrunning/face-parsing.PyTorch.git
```
※ face_utils.py は、隣接ディレクトリにある face-parsing.PyTorch を参照するように記述されています。🚀 使用方法 (Usage)基本的な実行コマンド例:Bash
```
python img2img.py \
  --input_dir data/input_images \
  --output_dir results/experiment_v1 \
  --config configs/latent-diffusion/ffhq-ldm-vq-4.yaml \
  --ckpt models/ldm/ffhq256/model.ckpt \
  --face_model_path models/face_parsing/79999_iter.pth \
  --snr -15.0 \
  --retransmission_rate 0.2 \
  --target_methods all
```

主な引数引数説明--snr通信路のSN比 (dB)。値が小さいほどノイズが強くなります。-r, --retransmission_rate再送予算 (0.0 〜 1.0)。画像の何割のピクセルを再送するかを指定します。--target_methods比較する手法を指定。all で全手法を実行。--struct_alphaStructural手法での不確実性とランダムノイズの混合比。--hybrid_alpha / --betaHybrid手法での不確実性とエッジ情報の重み付け。📊 比較手法 (Benchmarks / Target Methods)--target_methods で指定可能な戦略一覧です。メソッド名説明structural構造的不確実性: 時間的な不確実性マップを平滑化し、構造的な欠損を優先します。rawRaw Uncertainty: 拡散モデルのサンプリング過程で得られる生の不確実性を使用します。wacvWACV Benchmark: 既存研究に基づく不確実性評価指標を使用したベースライン。semanticSemantic Weighted: 受信側の不確実性と、送信側のセマンティック重要度（目・鼻等）を掛け合わせます。semantic_onlySemantic Only: 不確実性を無視し、単純に顔の重要パーツのみを再送する比較用手法。edge_recEdge (Rec): 再構成画像から抽出したエッジ周辺を再送します。edge_gtEdge (GT): Oracle手法。正解画像のエッジを使用する理論的上限値です。hybridHybrid: 構造的不確実性とエッジ情報を統合した戦略です。randomRandom: 領域をランダムに選択するベースラインです。📈 評価指標 (Metrics)PSNR: 物理的な画素値の再現性。LPIPS / DISTS: 人間の知覚に近い画質評価。ID Loss: FaceNetによる顔のアイデンティティ保持率の評価。FID: 生成画像群の統計的な自然さの評価。