# FossilDINOv2

このリポジトリは，DINOv2 の凍結特徴を用いたシンプルな多視点ベースラインを再現・可視化するためのものであり，クラスタリング側を調整して望ましい結果を得ることを目的としない。

## 研究目的
本研究の主張は次の1点です。

- **「多視点レンダ + DINOv2 凍結特徴 + 単純な統合 + 教師なしクラスタリング」で，化石・比較形態標本の群構造がどこまで見えるかを検証する。**

本リポジトリは精度競争よりも，以下を重視します。
- どこまで粗い群構造が見えるか
- どこから人間分類とずれるか
- 埋め込みとクラスタリングの根拠を可視化できるか

## 主解析（固定パイプライン）
1. 3D標本（ply/obj/stl/off）
2. 多視点レンダ
3. DINOv2 凍結特徴抽出
4. 視点統合（標準: mean pooling）
5. 標本埋め込み生成
6. 固定設定HDBSCAN（探索なし）
7. 可視化・評価（評価は後付け）

> 重要: 主解析では **best config search をしません**。ラベルは評価専用です。

## ディレクトリ構成
```text
project_root/
  data/
    meshes/
    renders/
    features/
    embeddings/
  results/
  configs/
    baseline.yaml
    visualization.yaml
  legacy/
    ... 旧実験コード（主張に使わない）
  src/
    render_multiview.py
    extract_features.py
    pool_embeddings.py
    cluster_baseline.py
    cluster_recursive_hdbscan.py
    cluster_branch_detector.py
    evaluate_with_labels.py
    visualize_embedding_space.py
    explain_vit_attention.py
    plot_hdbscan_trees.py
    report_neighbors.py
    utils/
      io.py
      geometry.py
      vision.py
      explain.py
  RESEARCH_SCOPE.md
  requirements.txt
```

## インストール
```bash
pip install -r requirements.txt
```

## 主解析の実行例
1. レンダ
```bash
python -m src.render_multiview --in data/meshes --out data/renders --views 12 --size 518 --auto-zoom --auto-zoom-probes 12
```

レンダ時の主なCLI補足:

- `--appearance {gray_lit,color_lit}`（既定: `gray_lit`）
  - `gray_lit`: 固定グレー材質 + 固定照明。標本ごとの色・テクスチャ差を抑え，形態由来の陰影/輪郭を比較したいときに使います。
  - `color_lit`: 頂点色やテクスチャがある場合は可能な限り反映しつつ，同じ固定照明で描画します（色情報が無い標本は安全に gray fallback）。
- `--auto-zoom`: 標本ごとにカメラ半径を自動調整し，投影サイズのばらつきを抑えます。
- `--target-fill-min` / `--target-fill-max`: `--auto-zoom` 時の目標充填率レンジを指定します。

appearanceを明示する例:

```bash
python -m src.render_multiview --in data/meshes --out data/renders_gray --views 24 --size 768 --appearance gray_lit
python -m src.render_multiview --in data/meshes --out data/renders_color --views 24 --size 768 --appearance color_lit
```

2. DINOv2 凍結特徴抽出
```bash
python -m src.extract_features --renders data/renders --out data/features --model dinov2_vits14 --device auto --image-size 518 --crop-size 518
```

3. 視点統合
```bash
python -m src.pool_embeddings --features data/features --out data/embeddings --pool mean
```

4. 固定設定クラスタリング(cluster_default_l2_eom_mcs10)
```bash
python -m src.cluster_baseline --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --out results/baseline_cluster
```

`src.cluster_baseline` の設定は CLI 引数で指定できます（デフォルト値は固定設定と同じです）。主な引数は以下です。

| 引数 | 型 / 選択肢 | デフォルト | 説明 |
|---|---|---:|---|
| `--emb` | `Path` | 必須 | 埋め込み `.npy` ファイル（または探索開始パス） |
| `--ids` | `Path` | 必須 | 標本 ID 一覧ファイル |
| `--out` | `Path` | 必須 | 出力ディレクトリ |
| `--normalize` | `none` / `l2` | `l2` | クラスタリング前の正規化方法 |
| `--method` | `hdbscan` | `hdbscan` | 手法（現状 `hdbscan` のみ） |
| `--metric` | `euclidean` | `euclidean` | 距離尺度 |
| `--min_cluster_size` | `int` | `10` | HDBSCAN の最小クラスタサイズ |
| `--min_samples` | `int` | `None` | HDBSCAN の `min_samples`（未指定時はライブラリ既定挙動） |
| `--selection_method` | `eom` / `leaf` | `eom` | HDBSCAN のクラスタ選択法 |

固定設定（`cluster_default_l2_eom_mcs10`）を明示する場合の実行例:

```bash
python -m src.cluster_baseline \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --out results/baseline_cluster \
  --metric euclidean \
  --normalize l2 \
  --min_cluster_size 10 \
  --min_samples 5 \
  --selection_method eom
```

### HDBSCAN tree 上の局所構造を拾うための補助方針

固定設定 `cluster_default_l2_eom_mcs10` は、DINOv2 埋め込み空間に対する最小限の HDBSCAN baseline です。一方で、HDBSCAN の single linkage tree / condensed tree には、EOM の最終ラベルでは巨大クラスタに吸収されるものの、局所的には有意味に見える枝・谷・高密度部分構造が現れる場合があります。

この補助比較は、クラスタリング結果を都合よく調整するためのものではありません。tree を目視して恣意的に選ぶのではなく、あらかじめ決めた固定ルールに基づいて局所構造を抽出し、分類群との対応を後から評価できるかを検証するためのものです。ラベルは split 条件や採用条件の決定には使わず、評価専用にします。

優先順位は以下とします。

```text
第一優先:
  selection_method="leaf"

第二優先:
  recursive HDBSCAN

第三優先:
  BranchDetector

第四優先:
  独自 valley-score 抽出
```

#### 1. leaf selection

EOM は安定した大きなクラスタを選びやすいため、巨大クラスタの内部にある小さく均質な構造を吸収する場合があります。`selection_method="leaf"` は、condensed tree の葉に近い細かなクラスタを選びやすく、EOM で mixed cluster が大きく残る場合の第一候補とします。

```bash
python -m src.cluster_baseline \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --out results/cluster_leaf \
  --metric euclidean \
  --normalize l2 \
  --min_cluster_size 10 \
  --min_samples 5 \
  --selection_method leaf
```

ただし、leaf は過分割しやすいため、`n_clusters`, `noise_ratio`, `largest_cluster_fraction`, `cluster_purity`, `ARI / NMI`, category-wise purity, 代表画像または近傍例を確認します。

#### 2. recursive HDBSCAN

recursive HDBSCAN は、全体に HDBSCAN をかけた後、大きすぎる mixed cluster だけを対象に、その内部で再度 HDBSCAN を実行する補助方針です。全クラスタを無差別に再分割するのではなく、split 対象を教師なしの固定ルールで限定します。

split 対象は以下とします。

```text
split対象:
  cluster_size > max(全体の20%, median_non_noise_cluster_size × 5)
```

この条件は次を意図しています。

- 全体の 20% を超える巨大クラスタは、複数カテゴリを吸収している可能性が高い
- 非ノイズクラスタの中央値の 5 倍を超えるクラスタは、他クラスタと比較して過大であり、内部構造を持つ可能性がある
- これらはラベルを使わない教師なし条件である

子クラスタ採用条件は以下とします。

```text
子クラスタ採用条件:
  - 子クラスタ数 >= 2
  - 最大子クラスタ比率 < 0.8
  - ノイズ率 < 0.5
  - largest_child_fraction が下がる
  - 子クラスタの silhouette / intra-distance が改善
  - ノイズ率が増えすぎない
  - 小さすぎるクラスタを量産しない
```

ラベルありデータでは、採用後に以下を確認します。

```text
ラベルあり評価での確認:
  - 親クラスタより cluster purity が改善するか
  - ARI / NMI が悪化しないか
  - category-wise purity が改善するか
```

ラベルは recursive split の採用判定には使わず、評価専用にします。
補助解析（recursive / BranchDetector）も `src.cluster_baseline` から引数で切り替えて実行します。

実行例:

```bash
python -m src.cluster_baseline \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --out results/cluster_recursive \
  --metric euclidean \
  --normalize l2 \
  --min_cluster_size 10 \
  --min_samples 5 \
  --selection_method leaf \
  --auxiliary_method recursive \
  --max_depth 1
```

#### 3. BranchDetector

BranchDetector は、HDBSCAN 後のクラスタ内部にある branch / branching hierarchy を検出するための後処理です。recursive HDBSCAN が巨大クラスタ内部を再度「密度クラスタリング」で分ける方法であるのに対し、BranchDetector はクラスタ内部の枝分かれ構造を検出する点が異なります。

実行例:

```bash
python -m src.cluster_baseline \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --out results/cluster_branches \
  --metric euclidean \
  --normalize l2 \
  --min_cluster_size 10 \
  --min_samples 5 \
  --selection_method eom \
  --auxiliary_method branch \
  --branch_min_cluster_size 10 \
  --branch_selection_method eom
```

| 方法 | 何を拾うか | 向いている状況 | 注意点 |
| --- | --- | --- | --- |
| leaf | condensed tree の細かい葉クラスタ | EOM が巨大クラスタを拾いすぎる場合 | 過分割しやすい |
| recursive HDBSCAN | 巨大クラスタ内部の局所密度クラスタ | mixed cluster を再分割したい場合 | split条件と採用条件を固定する必要 |
| BranchDetector | クラスタ内部の枝分かれ構造 | 密度的にはつながるが branch 構造を持つ場合 | category-like cluster 抽出とは目的が少し異なる |
| valley-score | single linkage tree 上の谷 | tree の局所的な凹みを明示的に拾いたい場合 | 独自指標なので説明責任が重い |

BranchDetector は主解析には入れず、クラスタ内部構造の補助的・探索的な比較対象として扱います。
`hdbscan` のバージョンによって `BranchDetector` が未実装の場合は、`src.cluster_branch_detector` はアップグレードを促す明確なエラーを返します。

#### 4. 独自 valley-score 抽出

single linkage tree 上の局所的な「谷」を明示的に拾う最終手段として、以下のような候補スコアを検討します。

```text
candidate cluster:
  - size >= min_cluster_size
  - internal_height が低い
  - parent_merge_height が高い
  - gap = parent_merge_height - internal_height が大きい
  - score = gap × log(size)
```

これは tree 上の凹みを定量化する独自指標の案であるため、主解析ではなく将来拡張または補助解析として扱います。

5. 埋め込み空間可視化
```bash
python -m src.visualize_embedding_space --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --clusters results/baseline_cluster/clusters.csv --format both --out results/vis
```
`results/vis/embedding_space_<method>_3d.html` をブラウザで開くと，マウスで回転・拡大縮小できる 3D インタラクティブ可視化を確認できます。

6. ViT 根拠可視化
```bash
python -m src.explain_vit_attention --renders data/renders --features data/features --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --clusters results/baseline_cluster/clusters.csv --out results/explain --image-size 518 --crop-size 518 --num-show 12
```

7. HDBSCAN tree 可視化
```bash
python -m src.plot_hdbscan_trees --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --clusters results/baseline_cluster/clusters.csv --out results/trees --selection_method eom --single_linkage_truncate_mode lastp --single_linkage_p 30
```

補助解析の結果を重ねて確認する場合は、`--clusters` に補助解析の `clusters.csv` を渡します（tree 自体は指定した HDBSCAN 設定から再計算されます）。

```bash
# leaf
python -m src.plot_hdbscan_trees --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --clusters results/cluster_leaf/clusters.csv --out results/trees_leaf --selection_method leaf --single_linkage_truncate_mode lastp --single_linkage_p 30

# recursive HDBSCAN の結果を重ねる例
python -m src.plot_hdbscan_trees --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --clusters results/cluster_recursive/clusters.csv --out results/trees_recursive --selection_method leaf --single_linkage_truncate_mode lastp --single_linkage_p 30

# BranchDetector の subgroup を重ねる例
python -m src.plot_hdbscan_trees --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --clusters results/cluster_branches/clusters.csv --out results/trees_branch --selection_method eom --single_linkage_truncate_mode lastp --single_linkage_p 30
```

8. ラベルあり評価（検証のみ）
```bash
python -m src.evaluate_with_labels --clusters results/baseline_cluster/clusters.csv --labels labels.txt --out results/baseline_cluster/eval
```

## 可視化の位置づけ
- `visualize_embedding_space.py`: PCA/UMAP は**可視化専用**（クラスタ最適化に使わない）
- `plot_hdbscan_trees.py`: tree は**説明専用**（設定選択に使わない）
- `explain_vit_attention.py`: attention rollout は**埋め込み形成に寄与した視覚的手がかり**の可視化

## 失敗時の解釈
結果が十分でない場合でも，
「DINOv2 の汎用視覚特徴は粗い形態差には反応するが，比較形態標本のクラスタリングにそのまま十分な表現空間を与えない可能性」
を示す意味があります。

## legacy について
`legacy/` 配下は旧実験コードです。主解析・主張の根拠には使いません。
