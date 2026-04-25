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
python -m src.render_multiview --in data/meshes --out data/renders --views 12 --size 518 --auto-zoom
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

4. 固定設定クラスタリング
```bash
python -m src.cluster_baseline --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --out results/baseline_cluster
```

5. 埋め込み空間可視化
```bash
python -m src.visualize_embedding_space --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --clusters results/baseline_cluster/clusters.csv --format both --out results/vis
```
`results/vis/embedding_space_<method>_3d.html` をブラウザで開くと，マウスで回転・拡大縮小できる 3D インタラクティブ可視化を確認できます。

6. ViT 根拠可視化
```bash
python -m src.explain_vit_attention --renders data/renders --features data/features --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --clusters results/baseline_cluster/clusters.csv --out results/explain --image-size 518 --crop-size 518 --num-show 24
```

7. HDBSCAN tree 可視化
```bash
python -m src.plot_hdbscan_trees --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --clusters results/baseline_cluster/clusters.csv --out results/trees --single_linkage_truncate_mode lastp --single_linkage_p 30
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
