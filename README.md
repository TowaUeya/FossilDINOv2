# FossilViT

FossilViT は、**3D標本から形態的な類似性を埋め込み空間として定量化する**ためのパイプラインです。  
具体的には、3Dメッシュを多視点レンダし、凍結した DINOv2（ViT）で画像特徴を抽出し、視点方向で統合して「標本ごとの埋め込みベクトル」を作ります。  
この埋め込み空間での距離（cosine/L2）を使って、(1) 類似標本検索 と (2) クラスタリング を行います。

> 要するに：
> - 3D形状 → 画像特徴 → 標本埋め込み
> - 埋め込み間の距離 = 分類形質（特徴量）の近さ
> - 距離に基づく検索・クラスタリングで、形態群の構造を解析

---

## パイプライン概要

1. **多視点レンダリング**（Open3D OffscreenRenderer）
   - 白背景・固定ライト
   - bboxでセンタリング + スケール正規化
   - 球面上の V 視点から画像生成
2. **特徴抽出**（凍結 DINOv2）
   - `torch.hub.load('facebookresearch/dinov2', model_name)`
   - `model.eval()` + `torch.inference_mode()` で推論のみ
   - 画像1枚ごとに特徴ベクトル `[D]`
3. **視点統合**（pooling）
   - 標本ごとに `[V, D]` を mean/max pooling
   - 最終的に標本埋め込み `[D]`
4. **活用**
   - 類似検索（NearestNeighbors）
   - クラスタリング（HDBSCAN推奨、KMeans任意）

---

## 用語（短い定義）

- **多視点レンダ**: 3D標本を複数方向から2D画像化する処理。
- **DINOv2 (ViT)**: ラベルなし学習済みの視覚Transformer。凍結特徴抽出器として使う。
- **特徴ベクトル `[D]`**: 画像1枚を D 次元で表現した数値ベクトル。
- **標本特徴 `[V, D]`**: 1標本の V 視点分の特徴を積んだ行列。
- **視点統合（pooling）**: `[V, D]` を1本の `[D]` にまとめる処理（mean がデフォルト）。
- **埋め込み（embedding）**: 標本を表す最終ベクトル。距離計算・検索・クラスタリングの入力。
- **HDBSCAN**: 密度ベースのクラスタリング。外れ点をノイズにできる。
- **ノイズ `-1`**: HDBSCAN が「どのクラスタにも属しにくい」と判断した点。
- **PCA `--pca 0.95`**: 累積寄与率95%を満たす最小次元数を自動選択する設定。
- **寄与率レポート（`--pca_report`）**: PCAで各主成分がどれだけ分散を説明したかのCSV。
- **UMAP**: クラスタリング補助の低次元化。密度構造を作りやすいが空間を歪めることがある。

---

## インストール

```bash
pip install -r requirements.txt
```

---

## 実行例（CLI）

以下は **必須 / 推奨 / 任意** の順で記載しています。

### 1) 必須（最低限、埋め込み生成まで）

```bash
python -m src.render_multiview --in data/meshes --out data/renders --views 24 --size 768
python -m src.extract_features --renders data/renders --out data/features --model dinov2_vits14 --device auto
python -m src.pool_embeddings --features data/features --out data/embeddings --pool mean
```

### 2) 推奨（類似検索）

```bash
QUERY_ID=$(head -n 1 data/embeddings/ids.txt)
python -m src.search --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --query "$QUERY_ID" --topk 10 --metric cosine --out results
```

次のコマンドは、カテゴリ階層を維持したまま標本ごとのCSVを `data/knn_results/` に保存し、実行ログも同ディレクトリに出力します。

```bash
python -m src.search_all \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --topk 10 \
  --metric cosine \
  --out data/knn_results
```

出力例:

- `data/knn_results/search_all.log`
- `data/knn_results/ammonite/knn_ammonite_0001.csv`
- `data/knn_results/trilobite/knn_trilobite_0012.csv`

カテゴリごとに「top-k近傍のカテゴリ一致率」を集計したい場合は、`search_all.log` を除外して `knn_*.csv` のみを読み取る次のコマンドを利用できます。

```bash
python -m src.knn_category_stats \
  --knn_dir data/knn_results \
  --out results/knn_category_summary.csv \
  --per_query_out results/knn_per_query_match_rate.csv
```

`results/knn_category_summary.csv` にはカテゴリごとに以下の統計量が出力されます。

- `n_queries`: そのカテゴリに属するクエリ標本数
- `mean_match_rate`: 一致率の平均（0〜1）
- `std_match_rate`, `var_match_rate`: 一致率の標準偏差・分散
- `min/q25/median/q75/max_match_rate`: 一致率の分布統計
- `mean_match_percent`, `std_match_percent`: パーセント表示の平均・標準偏差

### 3) 任意（クラスタリング）

#### まずはこれ（推奨設定）

```bash
python -m src.cluster --emb data/embeddings/embeddings.npy --out results --method hdbscan \
  --normalize l2 --metric cosine --pca 64 --min_cluster_size 10 --min_samples 1 \
  --cluster_selection_method leaf --pca_report results/pca_report.csv
```

#### 困ったら（UMAPを追加）

```bash
python -m src.cluster --emb data/embeddings/embeddings.npy --out results --method hdbscan \
  --normalize l2 --metric cosine --pca 50 --umap --umap_n_components 15 --umap_n_neighbors 30 --umap_min_dist 0.0 \
  --min_cluster_size 10 --min_samples 1 --cluster_selection_method eom
```

> 注意: UMAPは密度クラスタリングを助ける一方で、空間を加工します。  
> ModelNetのようなラベル付きデータでは、**ARI/NMI/purity で妥当性を必ず確認**してください。

### 4) 推奨（クラスタ設定の探索）

```bash
python -m src.cluster_sweep --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --out results/sweep
```

ラベルがある場合（推奨）:

```bash
python -m src.cluster_sweep --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --labels labels.txt --out results/sweep
```

---

## HDBSCANでノイズを減らす実践原則

このリポジトリのデフォルトは、次を重視します（蒸着は最終手段）。

1. **HDBSCANを保守的にしすぎない**
   - 典型的には `min_samples` を下げるとノイズ率は下がりやすい
   - 逆に上げるとノイズ率が増えやすい
2. **クラスタリング専用の低次元化を使う**
   - まず PCA 固定次元（例: 64, 50）
   - 必要に応じて UMAP を追加
3. **埋め込みは L2 正規化 + cosine を基本にする**

---

## PCA設定の注意（重要）

- `--pca 0.95` は、**累積寄与率95%を満たす最小次元数を自動採用**する指定です。
- ただしこれは「情報保持」の観点であり、**クラスタリング適性を保証しません**。
- 0.95で次元が大きく残ると、密度推定が難しくなり HDBSCAN に不利な場合があります。
- そのためクラスタリング用途では、まず `--pca 64` や `--pca 50` の固定次元を推奨します。
- `--pca_report` は必須ではないですが、過剰圧縮や寄与率の偏り検知に有効です。

---

## cluster_sweep の評価方針
`src.cluster_sweep` は、ノイズ率だけでなく以下を同時評価します。

- `noise_ratio`
- `mean_prob`（HDBSCANの所属確信度）
- `largest_cluster_fraction`（最大クラスタの非ノイズ点占有率）
- `giant_cluster_penalty`（巨大ごった煮クラスタへの罰則）
- ラベルがある場合: `ARI`, `NMI`, `purity`

`final_score` は巨大混合クラスタを不利にする設計です。
`best_config.yaml` と `best_clusters.csv` は `final_score` 最大の設定で出力されます。

---


## fine clustering と coarse clustering の使い分け

- **fine clustering（既存: `src.cluster_sweep`）**
  - HDBSCAN の最終ラベル（`labels_`）を対象に、`final_score` 最大化で設定探索。
  - 細分類（例: bone 内の dentary / vertebra など）に向く。
- **coarse clustering（改修: `src.coarse_cluster_sweep`）**
  - coarse mode は **tree の最大距離ギャップで分割を選ぶ方式ではありません**。  
    代わりに、HDBSCAN の粗い粒度設定（`min_cluster_size`, `selection_method` など）を再評価して `coarse_score` で選びます。
  - 既存の `sweep_results.csv`（`src.cluster_sweep` の結果）を読み、各設定を HDBSCAN で再実行して coarse 向き指標で再ランキングします。
  - `sweep_results.csv` が無い場合のみ、`cluster_sweep` と同等グリッド（`pca=[0,64,0.95]`, `min_cluster_size=[5,10,20,40]`, `min_samples=[1,2,5]`, `selection_method=[leaf,eom]`, `umap=[off,on]`）を内部で探索します。
  - 実験上、coarse 二分に効いたのは次の傾向です:
    - `min_cluster_size` が大きい
    - `selection_method=eom`
    - `noise_ratio` がほぼゼロ
    - `second_largest_cluster_fraction`（第2クラスタ比率）が十分大きい
  - そのため `coarse_score` は主に以下を使って best を選びます:
    - `second_largest_cluster_fraction`（最重要）
    - `effective_num_clusters≈2`
    - `size_entropy`
    - `noise_ratio`
    - `mean_prob_non_noise`
    - `min_cluster_size`（軽いボーナス）
    - `selection_method=eom`（軽いボーナス）
  - hard reject は `n_clusters<=1` / `noise_ratio>0.10` / `largest_cluster_fraction>0.70` / `second_largest_cluster_fraction<0.15` / `effective_num_clusters<1.8` です。
  - ラベル（`labels.txt`）は検証専用です。探索・再ランキング・設定選択には**一切使いません**。

### coarse mode 実行例

1) 自動（既存 sweep を coarse 再ランキング, 推奨）

```bash
python -m src.coarse_cluster_sweep \
  --sweep_csv results/sweep/sweep_results.csv \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --out results/coarse_sweep
```

2) 一覧表示（上位N件）

```bash
python -m src.coarse_cluster_sweep \
  --sweep_csv results/sweep/sweep_results.csv \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --out results/coarse_sweep \
  --list --topn 20
```

3) coarse_rank 指定で採用

```bash
python -m src.coarse_cluster_sweep \
  --sweep_csv results/sweep/sweep_results.csv \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --out results/coarse_sweep \
  --pick 10
```

4) ラベル付き診断（探索後の検証のみ）

```bash
python -m src.coarse_cluster_sweep \
  --sweep_csv results/sweep/sweep_results.csv \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --labels labels.txt \
  --out results/coarse_sweep
```

### coarse mode の出力

- `coarse_sweep_results.csv`
  - `coarse_rank`, `coarse_score`, `invalid`
  - `noise_ratio`, `n_clusters`, `largest_cluster_fraction`, `second_largest_cluster_fraction`
  - `size_entropy`, `effective_num_clusters`, `mean_prob_non_noise`
  - `pca`, `umap`, `min_cluster_size`, `min_samples`, `selection_method`
  - `ari`, `nmi`, `purity`（診断専用）
- `best_coarse_clusters.csv`（`specimen_id`, `cluster_id`, `prob`）
- `best_coarse_config.yaml`
- `summary.json`

## クラスタ分析レポート生成

best_clusters.csv を論文向けに診断するスクリプト:

```bash
python -m src.analyze_clusters --clusters results/sweep/best_clusters.csv --out results
```

ラベル付きの場合:

```bash
python -m src.analyze_clusters --clusters results/sweep/best_clusters.csv --labels labels.txt --out results
```

出力:

- `results/cluster_report.json`
- `results/cluster_purity.csv`
- `results/largest_clusters.csv`

---

## 出力ディレクトリ構成

```text
project_root/
  data/
    meshes/
    renders/
    features/
    embeddings/
  results/
  src/
    render_multiview.py
    extract_features.py
    pool_embeddings.py
    cluster.py
    cluster_sweep.py
    analyze_clusters.py
    search.py
  configs/default.yaml
  requirements.txt
```

---

## 補足

- 蒸着（ノイズ点を後段で強制割当）は、解析上の都合で最終手段としてのみ使用してください。
- まずは `min_samples`・PCA固定次元・UMAP有無の見直しで密度構造を改善するのが正攻法です。



## HDBSCANの説明図（dendrogram / icicle plot）

`cluster_sweep` の探索結果（`sweep_results.csv`）を唯一の設定ソースとして、
`single_linkage_tree_` / `condensed_tree_` を画像・CSVで出力できます。

引数なし（デフォルトパス + `final_score` 最大設定を自動選択）:

```bash
python -m src.plot_hdbscan_trees
```

探索結果の上位20件を番号付きで一覧:

```bash
python -m src.plot_hdbscan_trees --list --top 20
```

`rank=3` の設定で説明図を生成:

```bash
python -m src.plot_hdbscan_trees --pick 3
```

パスだけ指定（デフォルト以外）:

```bash
python -m src.plot_hdbscan_trees \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --sweep_csv results/sweep/sweep_results.csv \
  --out results/trees
```

主な出力:

- `results/trees/selected_config.yaml`（選択行 + 実行時グローバル設定 + パス）
- `results/trees/labels.csv`（`specimen_id, cluster_id, prob`）
- `results/trees/condensed_tree.png`
- `results/trees/condensed_tree_selected.png`
- `results/trees/single_linkage_tree.png`（`--skip_single` で省略可）
- `results/trees/single_linkage_tree_with_leaf_labels.png`（全リーフに3Dモデル名を表示）
- `results/trees/single_linkage_tree_with_leaf_labels.pdf`（全リーフに3Dモデル名を表示）
- `results/trees/single_linkage_tree_with_leaf_labels.html`（生成可能な場合のみ。対話的に拡大/移動可能）
- 可能なら木構造CSV: `condensed_tree.csv`, `single_linkage_tree.csv`

巨大データでフルラベル版が重すぎる場合は、次で無効化できます。

```bash
python -m src.plot_hdbscan_trees --skip_single_leaf_labels
```

形式の使い分け目安:

- **HTML**: 巨大dendrogramを拡大・移動しながら確認したいとき（第一候補）
- **PDF**: 共有・印刷向けの保存版
- **PNG**: 手早い確認向け

> 補足: `sweep_results.csv` で `umap=True` かつ `pca<=0` の設定が選ばれた場合、
> `cluster_sweep` と同じ挙動に合わせて内部的に `pca=50` を適用します。

## Optional prefilter: physical-size / coarse-shape based routing

この機能は **shape embedding 本線（render → DINOv2 → pooling）とは独立** したオプションです。  
既存の埋め込み生成や `src.search_all` / `src.cluster` の既定動作は変更せず、raw 3D から抽出したメタ特徴（物理サイズ・粗形状）で前段の候補集合を粗く分割します。

- 本線にはサイズ・色を混ぜません
- prefilter は「候補削減 / ルーティング」専用です
- 最終的な近傍順位・最終クラスタ判断は shape embedding 側で行います

### 何を作るか

1. `src.prefilter_metadata`
   - raw mesh/point cloud から metadata CSV を作成
   - **物理サイズ特徴** を抽出（coarse routing 用、厳密形状解析ではない）:
     - `surface_area`
     - `convex_hull_volume`（有効時のみ）
     - `convex_hull_area`
     - `bbox_longest`
     - `equiv_diameter_hull`
     - `is_watertight`（`--size_compute_mode full --enable_mesh_volume` のときのみ計算）
   - `--size_compute_mode fast` は AABB ベースのみ（watertight 判定・mesh volume は未計算）
   - `--size_compute_mode full --enable_mesh_volume` のときのみ watertight 判定を行い、`mesh_volume` / `equiv_diameter_mesh` を算出
   - volume の階層 fallback（`mesh_volume` → `hull_volume` → `surface_area` → `bbox_longest`）は full モード時のみ有効
   - PCA軸長ベースの連続量を併せて出力:
     - `flatness_ratio = pc3_length / pc1_length`（global flatness の近似）
     - `thinness_ratio = pc2_length / pc1_length`（global thinness の近似）
   - volume/hull 計算の安全ガードとして退化フラグを使用:
     - `is_planar_degenerate`（`flatness_ratio < planar_degenerate_thresh`）
     - `is_linear_degenerate`（`thinness_ratio < linear_degenerate_thresh`）
   - volume を止めるのは退化ケースのみ（「やや薄い」標本を一律除外しない）
   - convex hull は全点ではなく downsample 後に計算（`--hull_max_points`, `--hull_sampling`）
   - 粗形状特徴（`aspect_*`, `elongation_*`, `flatness_ratio`）を併用可能
   - 既定は `physical_bins` で全標本を coarse group `pregroup_id` に付与（noiseを作らない）
   - `hdbscan` は実験モードとして保持
2. `src.search_all_prefilter`
   - `src.search_all` 互換に近い出力形式で、prefilter 付き全件検索を実行
3. `src.cluster_prefilter`
   - `pregroup_id` ごとに分割して shape embedding クラスタリングを実行
4. `src.evaluate_prefilter`
   - `Fossil_category.csv` のラベルを使って prefilter の purity / 候補削減率を評価

### 推奨方針（重要）

- 推奨 `use_color` は **`off`**（色は補助実験用）
- 推奨 `grouping_method` は **`physical_bins`**
- prefilter の目的は **クラスタ発見ではなく coarse routing**
- HDBSCAN prefilter は **実験用**（noise が増える可能性あり）
- prefilter の評価は purity だけでは不十分です
- **same-group recall / same+adjacent-group recall** と、**off / soft / strict の最終検索精度**を併せて確認してください
- これにより routing の候補保持率と、最終順位品質への影響を同時に判断できます
- 最終順位・最終クラスタ判断は必ず shape embedding 側で実施
- デフォルトは **AABB ベース高速特徴**（`bbox_longest`, `log_bbox_longest`, `bbox_volume`, `log_bbox_volume`）を主軸にします
- `--size_compute_mode fast` では mesh volume 計算と watertight 判定を行いません
- hull/mesh volume は `--size_compute_mode full` かつ明示フラグ有効時のみ計算する optional 機能です
- `log_*` 系は `log10(max(value, 1e-12))` で安全に計算します
- coarse routing の shape split では `--shape_split flatness_ratio` も利用可能

### 色特徴について

- 色は optional です。
- STL は標準仕様上、通常は色を持ちません。
- PLY は頂点色を持つ場合があります。
- OBJ の色はファイル依存で、常に保証されません。
- 色が無い標本でも処理は継続し、`has_color=0` として扱います。

### 推奨運用

```bash
# 1) 物理量ベース prefilter
python -m src.prefilter_metadata \
  --in data/meshes \
  --out data/prefilter \
  --grouping_method physical_bins \
  --use_color off \
  --size_compute_mode fast \
  --shape_split none \
  --jobs 4

# 1-b) flatness で coarse routing を分割する例
python -m src.prefilter_metadata \
  --in data/meshes \
  --out data/prefilter \
  --grouping_method physical_bins \
  --use_color off \
  --size_compute_mode full \
  --enable_hull_features \
  --enable_mesh_volume \
  --size_source auto \
  --volume_bins 5 \
  --hull_max_points 3000 \
  --hull_sampling random \
  --shape_split flatness_ratio \
  --planar_degenerate_thresh 0.02 \
  --linear_degenerate_thresh 0.02 \
  --jobs 4 \
  --progress_every 5

# NOTE: 標本ごとの timing ログが INFO/WARNING で出力されます（`--slow_threshold_sec` のデフォルトは 60.0 秒）。

# 2) 検索結果の生成（off / soft / strict を別ディレクトリに保存）
python -m src.search_all_prefilter \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --prefilter_csv data/prefilter/metadata_features.csv \
  --prefilter_mode off \
  --topk 10 \
  --metric cosine \
  --out data/knn_results_off

python -m src.search_all_prefilter \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --prefilter_csv data/prefilter/metadata_features.csv \
  --prefilter_mode soft \
  --expand_strategy adjacent \
  --candidate_multiplier 2.0 \
  --topk 10 \
  --metric cosine \
  --out data/knn_results_soft

python -m src.search_all_prefilter \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --prefilter_csv data/prefilter/metadata_features.csv \
  --prefilter_mode strict \
  --topk 10 \
  --metric cosine \
  --out data/knn_results_strict

# 3) 前段 pregroup ごとのクラスタリング
python -m src.cluster_prefilter \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --prefilter_csv data/prefilter/metadata_features.csv \
  --method hdbscan \
  --normalize l2 \
  --metric cosine \
  --pca 64 \
  --min_cluster_size 10 \
  --min_samples 1 \
  --cluster_selection_method leaf \
  --out results/prefilter_cluster

# 4) prefilter purity 評価（coarse routing の健全性確認）
python -m src.evaluate_prefilter \
  --prefilter_csv data/prefilter/metadata_features.csv \
  --labels_csv data/prefilter/Fossil_category.csv \
  --out results/prefilter_eval

# 5) off の検索結果評価（Fossil category ラベルで最終検索精度を集計）
python -m src.evaluate_knn_prefilter \
  --knn_dir data/knn_results_off \
  --labels_csv data/prefilter/Fossil_category.csv \
  --out results/knn_eval_off \
  --topk_values 1,5,10

# 6) soft の検索結果評価 + prefilter recall
python -m src.evaluate_knn_prefilter \
  --knn_dir data/knn_results_soft \
  --labels_csv data/prefilter/Fossil_category.csv \
  --prefilter_csv data/prefilter/metadata_features.csv \
  --out results/knn_eval_soft \
  --topk_values 1,5,10

# 7) strict の検索結果評価 + prefilter recall
python -m src.evaluate_knn_prefilter \
  --knn_dir data/knn_results_strict \
  --labels_csv data/prefilter/Fossil_category.csv \
  --prefilter_csv data/prefilter/metadata_features.csv \
  --out results/knn_eval_strict \
  --topk_values 1,5,10

# 8) off / soft / strict の横並び比較
python -m src.compare_knn_eval \
  --baseline_dir results/knn_eval_off \
  --soft_dir results/knn_eval_soft \
  --strict_dir results/knn_eval_strict \
  --out results/knn_eval_compare
```

### texture/color 補助枝（optional）

shape render 出力を再利用して、軽量な色ヒストグラム特徴を標本単位で作成できます。  
この枝は **補助信号** であり、shape embedding 本線そのものは変更しません。

```bash
python -m src.texture_features \
  --renders data/renders \
  --out data/texture_features \
  --color_space lab \
  --bins 16 \
  --pool mean
```

### shape first, rerank later（late fusion 検索）

`src.search_all_fusion` は、まず shape embedding のみで候補を取り、top候補内だけ size/texture を線形結合して再ランクします。

\[
D_{final}(q,x)=D_{shape}(q,x)+\lambda_{size}D_{size}(q,x)+\lambda_{tex}D_{tex}(q,x)
\]

```bash
python -m src.search_all_fusion \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --prefilter_csv data/prefilter/metadata_features.csv \
  --topk 10 \
  --rerank_topk 50 \
  --lambda_size 0.05 \
  --size_feature apparent_size_proxy \
  --size_penalty_mode ratio_penalty \
  --size_distance_norm robust \
  --prefilter_mode soft \
  --expand_strategy adjacent \
  --metric cosine \
  --out data/knn_results_fusion_apparent_size
```

- `--prefilter_mode off` なら全体探索にフォールバック。
- `--prefilter_csv` 未指定時は size/routing を使わず、shape(+texture)のみで動作。
- `--texture_features` 未指定時は texture 距離を使わず、shape(+size)のみで動作。
- `--size_feature` で size 距離の列を切り替え可能（`bbox_longest`, `log_bbox_longest`, `bbox_volume`, `log_bbox_volume`, `apparent_size_proxy`, `apparent_size_volume_proxy` など）。
- `--size_penalty_mode` は `plain_distance / ratio_penalty / margin_gate` を切り替え可能（既定: `ratio_penalty`）。
- `--size_distance_norm` で size 距離の正規化を指定可能（`none`, `zscore`, `minmax`, `robust`、既定は `robust`）。
- 既定のクラスタリング本線（`src.cluster`）は shape-only のままです。
- fused clustering は主流ではなく実験用途として扱ってください。

評価（top-k 一致率の比較）:

比較評価（同一 166 query 集合で `shape_only` と `shape+size` を横並び）を行う場合は、次の順で実行します。

1. `src.search_all_fusion` を `shape_only` 設定で実行し、`knn_all_fusion.csv` を保存
2. `src.search_all_fusion` を `shape+size` 設定で実行し、`knn_all_fusion.csv` を保存
3. `src.combine_fusion_runs` で 2 つ（必要なら 3 つ）の CSV を `fusion_mode` 付きで結合
4. `src.evaluate_fusion` で結合後 CSV を評価

```bash
python -m src.combine_fusion_runs \
  --shape_only_csv data/knn_results_fusion_shape_only/knn_all_fusion.csv \
  --shape_size_csv data/knn_results_fusion_shape_size/knn_all_fusion.csv \
  --out data/knn_results_fusion_compare/knn_all_fusion_compare.csv
```

```bash
python -m src.evaluate_fusion \
  --knn_csv data/knn_results_fusion_compare/knn_all_fusion_compare.csv \
  --labels_csv data/prefilter/Fossil_category.csv \
  --out results/fusion_eval_compare \
  --strict_query_set_check
```

単一 run の CSV をそのまま評価する場合（従来どおり）:

```bash
python -m src.evaluate_fusion \
  --knn_csv data/knn_results_fusion/knn_all_fusion.csv \
  --labels_csv data/prefilter/Fossil_category.csv \
  --out results/fusion_eval
```

late fusion の次段として、size signal と `lambda_size` を系統比較する `src.run_fusion_ablation` を追加しています（shape 本線は untouched）。

```bash
python -m src.run_fusion_ablation \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --prefilter_csv data/prefilter/metadata_features.csv \
  --labels_csv data/prefilter/Fossil_category.csv \
  --topk 10 \
  --rerank_topk 50 \
  --lambda_grid 0.02,0.05,0.1,0.2 \
  --size_features bbox_longest,log_bbox_longest,log_bbox_volume,apparent_size_proxy,apparent_size_volume_proxy \
  --size_penalty_modes plain_distance,ratio_penalty,margin_gate \
  --prefilter_mode soft \
  --expand_strategy adjacent \
  --out results/fusion_ablation_apparent_size
```

主な出力:
- `fusion_ablation_overall.csv`
- `fusion_ablation_per_category.csv`
- `delta_vs_shape_only.csv`
- `delta_vs_shape_only_per_category.csv`
- `best_fusion_config.json`

### 既存検索 / 既存クラスタとの違い

- `src.search_all` / `src.cluster` は従来どおり利用可能（変更なし）。
- 新機能は別CLIでのみ有効化され、使わない限り既存ワークフローに影響しません。
- `search_all_prefilter` の `--prefilter_mode off` は全体探索にフォールバックします。

### 主な出力

- `data/prefilter/metadata_features.csv`
  - `specimen_id`, `source_path`, `has_color`
  - `size_x/size_y/size_z`, `max_extent`, `bbox_longest`, `bbox_volume`
  - `log_bbox_longest`, `log_bbox_volume`
  - `surface_area`, `convex_hull_volume`, `convex_hull_area`
  - `is_watertight`, `mesh_volume`, `equiv_diameter_hull`, `equiv_diameter_mesh`
  - `pc1_length`, `pc2_length`, `pc3_length`, `flatness_ratio`, `thinness_ratio`
  - `size_metric`, `size_scalar`, `log_size_scalar`
  - `is_planar_degenerate`, `is_linear_degenerate`
  - `is_planar_like`, `is_linear_like`, `n_points_original`, `n_points_used_for_hull`
  - `hull_failed`, `volume_failed`
  - `aspect_xy/aspect_xz/aspect_yz`, `elongation_12/13/23`
  - `mean_r/g/b`, `std_r/g/b`
  - `size_volume`, `log_size_volume`, `volume_bin`, `shape_flag`, `grouping_method`
  - `pregroup_id`, `pregroup_prob`
  - `mesh_load_sec`, `aabb_sec`, `hull_sec`, `mesh_volume_sec`, `total_sec`（処理時間ログ）
- `data/prefilter/pregroup_summary.json`
  - `n_planar_degenerate`, `n_linear_degenerate`
  - `n_flatness_lt_0_02/0_05/0_10`, `n_thinness_lt_0_02/0_05/0_10`
  - `n_planar_like`, `n_linear_like`, `size_metric_counts`
  - `n_hull_failed`, `n_volume_failed`, `mean_points_used_for_hull`
  - timing 集計: `mean_mesh_load_sec`, `mean_aabb_sec`, `mean_hull_sec`, `mean_mesh_volume_sec`, `mean_total_sec`, `max_total_sec`
  - 遅延標本情報: `slowest_specimen_id`, `n_slow_specimens`, `slow_threshold_sec`
- `data/prefilter/pregroup_counts.csv`
- `data/prefilter/run_config.yaml`
- `results/prefilter_cluster/clusters_prefilter.csv`
- `results/prefilter_cluster/cluster_prefilter_summary.json`
- `results/prefilter_cluster/cluster_prefilter_config.yaml`
- `results/prefilter_eval/prefilter_eval_summary.json`
- `results/prefilter_eval/prefilter_eval_groups.csv`
- `results/knn_eval_*/knn_eval_per_query.csv`
- `results/knn_eval_*/knn_eval_per_label.csv`
- `results/knn_eval_*/knn_eval_summary.json`
- `results/knn_eval_*/prefilter_recall_per_query.csv`（`--prefilter_csv` 指定時）
- `results/knn_eval_*/prefilter_recall_per_label.csv`（`--prefilter_csv` 指定時）
- `results/knn_eval_*/prefilter_recall_summary.json`（`--prefilter_csv` 指定時）
- `results/knn_eval_compare/compare_overall.csv`
- `results/knn_eval_compare/compare_per_label.csv`
- `data/texture_features/texture_features.npy`
- `data/texture_features/ids.txt`
- `data/texture_features/texture_feature_config.yaml`
- `data/knn_results_fusion/knn_all_fusion.csv`
