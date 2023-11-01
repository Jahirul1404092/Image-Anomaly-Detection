# 設定ファイル（config.yaml）
ここでは、設定ファイルのさまざまな属性について説明します。  
設定ファイルは通常、`hamacho/plug_in/{model}/config.yaml`の場所に配置されます。しかし、これは`CUI` からも渡すこともできます。この`yaml`ファイルには、次の上位属性が含まれます。

- `dataset` : 画像データに関する情報が記載されています。(フォルダ構成、学習時の処理画像サイズ、前処理等) 
- `model` : モデルに関連する設定可能な項目の多くのサブ属性を提供します。（バックボーン名、サンプリング数等）
- `metrics` : 学習、評価、および推論時にしようされるメトリックのコレクションが含まれています。
- `project` : 評価と推論のための出力スタイルを制御するための属性がほとんど含まれており、ログを制御するためのブール属性が含まれています。
- `trainer` : `hamacho`コードは、主に`PyTorch-Lightning`に基づいています。この`trainer`クラス（およびその属性）は、主にコアAPIから取得されます。（ユーザーは特に触る必要はありません。）

----------------------------------------------------------
以下では、各上位属性について詳しく説明します。最も重要な属性について説明し、明らかなものはスキップします。ただし、モデルのタイプ（つまり、`padim`、`patchcore`など）に応じて、いくつかのモデル指向のサブ属性が異なる場合があることに注意してください。


## データ

```yaml
dataset:
  name: mvtec #options: [mvtec, btech, folder]
  format: mvtec
  path: ./datasets/MVTec
  task: segmentation
  category: bottle
  image_size: 224
  train_batch_size: 12
  test_batch_size: 12
  num_workers: 0
  normal_dir: null
  abnormal_dir: null
  normal_test_dir: null
  transform_config:
    train: null
    val: null
  create_validation_set: false
  tiling:
    apply: false
    tile_size: null
    stride: null
    remove_border_count: 0
    use_random_tiling: False
    random_tile_count: 16
```

**説明**  

- `name` : データ形式タイプを表す任意の生の文字列です。  
- `format` : サポートされるデータローダーを定義します。  
  現在、サポートされているデータローダーは、`mvtec`、`folder`および`filelist`フォーマットの３つです。  
      - `mvtec`フォーマットは、研究目的のMVTecデータセットのベンチマークに使用されます。  
  新しいモデルを追加する場合は、これらのデータセットで確認してください。
      - カスタムデータセットの場合、`folder`または`filelist`フォーマットを設定できます。
- `path` : データセットのルートディレクトリです。
      - `mvtec`と`folder`フォーマットのデータセットを選択した場合、このパス指定は必須です。
- `task` : `classfication`と`segmentation`の二つがサポートされています。  
  違いは出力形式です。
      - `classfication`が設定されている場合、出力結果は画像上に`normal??%` `abnormal??%`と正常か異常の判定結果のテキストを付けたヒートマップを出力します。計算結果については、画像レベルのみ表示されます。
      - `segmentation`が設定されている場合、出力結果はピクセルレベルで正常異常を判定したマスク画像が出力されます。計算結果については、学習時にマスクが与えられていない場合は画像レベルのみ表示されます。マスクが与えられている場合はピクセルレベルの計算結果も表示されます。   
-  `category`: 通常、`path`ディレクトリ内に配置されたフォルダ名です。製品IDや名称などを表します。  
  たとえば、カーペットオブジェクトから異常を検出する必要がある場合は、`category`を`carpet`と名付ける必要があります。
-  `num_workers` : データの前処理に使用されるCPUワーカーの数を指定します。  
  この値を適切に設定することで、データの読み込みと前処理を並列化し、モデルのトレーニング時間を短縮できます。ただし、適切な値を設定する必要があります。  
  ワーカーの数が多すぎると、データの読み込みと前処理のオーバーヘッドが増加し、システムリソースを消費します。一方、ワーカーの数が少なすぎると、ボトルネックが発生し、モデルのトレーニング時間が長くなる可能性があります。  
  通常、ワーカーの数はCPUのコア数に等しいか、その数以下に設定されます。ただし、メモリが不足している場合は、ワーカー数を減らす必要があるかもしれません。
- `transform_config`：このフラグを立てることで、画像のaugmentationを有効にすることができます。  
  （が`hamacho`には実装されていません。）
  モデルが複数のエポックで学習されている場合、少ないデータセットでもバリエーションを増やして学習することが可能です。
- `normal_dir` : 正常画像が格納されているフォルダ名です。
- `abnormal_dir` : 異常画像が格納されているフォルダ名です。
- `normal_test_dir` : テスト用に使う正常画像が格納されているフォルダ名です。
      - `normal_test_dir`は現状使うことができません。  
      - 詳細はこちらの[リンク](https://chowagiken.slack.com/archives/C031VDC258T/p1668662477809799)をご覧ください。
- `tiling`: 入力画像を縮小せずに、パッチに分割して学習する機能です。  
  `True`に設定すると、画像が複数のパッチに抽出されます。  
  どのような時に活用するかというと、ターゲットの異常が画像サイズに比べて小さすぎる場合、`tiling=True`にすることで、抽出された各パッチが個別に処理され、小さな異常も検出できることができます。（この機能はまだ未実装です。）  
      - `apply`: ブール値のフラグ、`true`または`false`で有効/無効にできます。
      - `tile_size`: タイルのサイズ。画像分割時のパッチのサイズ。
      - `stride`: 画像分割時、スライドするサイズ。`tile_size`より小さいと重なりができ、同じ値にすると重ならずに分割できる。
      - `remove_border_count`: タイリングの前に画像から削除するピクセル数。
      - `use_random_tiling`: ブール値のフラグ、`true`または`false`.
      - `random_tile_count`: 画像からサンプリングするランダムタイルの数。


深層学習に関する文献では、入力画像のサイズは通常、224～768ピクセルの範囲です。しかし、産業用途の場合、入力画像のサイズは通常、はるかに大きくなります。学習時、これらの画像はモデルに入力される前に縮小されます。ただし、異常個所が非常に小さい場合、異常検出時に問題になります。画像が縮小されると、異常個所が消失してしまい、検出が非常に困難になります。この問題に対処するための一般的な方法は、入力画像を分割して処理することで、リサイズ操作中に情報が失われないようにすることです。[詳細](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/100_datamodules/104_tiling.ipynb)

---

## モデル構成

```yaml
model:
  name: patchcore
  backbone: wide_resnet50_2
  layers:
    - layer2
    - layer3
  coreset_sampling_ratio: 0.1
  num_neighbors: 9
  weight_file: weights/trained_data.hmc
  normalization_method: min_max # options: [null, min_max, cdf]
```

**説明**

- `name`: サポートされているモデルの名前です。現在、`patchcore`と`padim`があります。ほとんどの場合、Patchcoreの方がPadimよりも優れていますが、Patch-coreはややメモリ効率が悪いです。
- `backbone`: バックボーンモデルです。サンプルの特徴表現を抽出するために使用されます。  
  両方のモデルは異なるバックボーンを使用します。
  Patchcoreでは、 `wide_resnet50_2`、Padimではデフォルトで`resnet18`です。  
  これらのバックボーンを変更することができますが、以下の属性のより良いオプションを見つけるためにいくつかのテストが必要になる場合があります。
  
- `layers`: バックボーンモデルからの特徴出力ベクトルの数。これらのすべての出力ベクトルは結合され、豊富な特徴表現を構築するために使用されます。 Padim（バックボーン： `resnet18`）では、レイヤーは `layers1、layers2、layers3` です。Patchcore（ `wide_resnet50_2`）では、レイヤーは `layer2、layer3` です。参考までに、レイヤーを選択する一般的なアイデアは、バックボーンの低〜中レベルの周辺から選択することです。高レベルレイヤーはImageNetデータセットにより傾向があります。
- `normalization_method`: モデルから受け取った生の異常マップスコアの出力に通常適用される正規化方法です。サポートされているオプションは、min-max、cdf（累積分布関数）、sigma6です。
      - `min-max`：異常スコアに対して適用される最小値〜最大値正規化方法。
      - `cdf`：累積分布関数（または **CDF**）。異常スコアの画像レベルとピクセルレベルを標準化する正規化方法の1つ。DOI：10.1109/TGRS.2014.2387574。
      - `sigma6`: この正規化手法は、モデルが悪いサンプルなしで訓練された場合に適用できます。特に、CUIで `--no-bad-mode` が有効になっている場合に使用します。
- `coreset_sampling_ration`: エンベディングをサブサンプリングするためのコアセットサンプリング率。 `patch-core` モデルに関連します。
- `num_neighbors`: 最近傍の数。 `patch-core` モデルにのみ関連します。
 

---

## メトリクス構成

```yaml
metrics:
  image:
    - F1Score
    - AUROC
    - BinaryRecall
    - BinaryAccuracy
    - BinaryPrecision
  pixel:
    - F1Score
    - AUROC
    - BinaryRecall
    - BinaryAccuracy
    - BinaryPrecision
  threshold:
    image_norm: 0.5 # must be within 0 and 1
    pixel_norm: 0.5 # must be within 0 and 1
    image_default: 0
    pixel_default: 0
    adaptive: true
```

**説明**

- `image`: モデルのパフォーマンスを評価するために使用されるメトリクスのリスト。
- `pixel`: モデルのパフォーマンスを評価するために使用されるメトリクスのリスト。このピクセルレベルのメトリックスを使用するには、バイナリマスクデータセットを不良画像のグラウンドトゥルースとして提供する必要があります。
- `threshold.adaptive`: Trueの場合、閾値値は自動で評価されます。これは、`F-1`スコアを最大化することによって行われます。
- `threshold.image_norm`: 推論時に、この属性は閾値値を微調整するために使用できます（0-1の範囲）。
- `threshold.pixel_norm`: `image_norm`と同様です。


---

## プロジェクト構成

```yaml
project:
  seed: 0
  path: ./results
  test_dir_name: test_predictions
  inference_dir_name: inference_results
  log_images_to: [local]
  logger: false # options: [tensorboard, csv] or combinations.
  save_outputs:
    test:
      image:
        segmentation:
          - input_image
          - histogram
          - ground_truth_mask
          - predicted_heat_map
          - grayscale
          - predicted_mask
          - segmentation_result
        classification:
          - input_image
          - histogram
          - prediction
      csv:
        - anomaly_map
        - metrics
    inference:
      image:
        segmentation:
          - input_image
          - histogram
          - ground_truth_mask
          - predicted_heat_map
          - grayscale
          - predicted_mask
          - segmentation_result
        classification:
          - input_image
          - histogram
          - prediction
      csv:
        - anomaly_map
        - metrics
    save_combined_result_as_image: true
    add_label_on_image: true
```

**説明**  

- `path`: ルートディレクトリに保存する場所を指定します。
- `log_images_to`: 外部ドキュメント[LINK](https://openvinotoolkit.github.io/anomalib/tutorials/logging.html)を参照してください。
- `save_outputs.image.segmentation`: データセットの評価を実行した後に、ユーザーが取得したい出力のリストが含まれています。  
基本的に、`task=segmentation`の場合、これらの出力はローカルディスクに保存されます。
- `save_outputs.image.classification`: 同様に、`task=classification`の場合のリスト。
- `save_outputs.image.csv`: 
    - `True`の場合、異常マップ（スコア）がCSV形式で保存されます。
- `save_combined_result_as_image`: 
    - `True`の場合、`save_outputs.image.{task-type}`の出力が一緒に保存されます。
- `add_label_on_image`: 
    - `True`の場合、予測されたラベルがテキストとして入力画像の上部に配置されます。  
      このフラグは、`Classification`の結果にのみ有効です。

---

## トレーナー構成

```python
# PL Trainer Args. Don't add extra parameter here.
trainer:
  accelerator: auto # <"cpu", "gpu", "tpu", "ipu", "hpu", "auto">
  accumulate_grad_batches: 1
  amp_backend: native
  auto_lr_find: false
  auto_scale_batch_size: false
  auto_select_gpus: false
  benchmark: false
  check_val_every_n_epoch: 1 # Don't validate before extracting features.
  default_root_dir: null
  detect_anomaly: false
  deterministic: false
  devices: 1
  enable_checkpointing: false
  enable_model_summary: true
  enable_progress_bar: true
  fast_dev_run: false
  gpus: null # Set automatically
  gradient_clip_val: 0
  ipus: null
  limit_predict_batches: 1.0
  limit_test_batches: 1.0
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  log_every_n_steps: 50
  # log_gpu_memory: null
  max_epochs: 1
  max_steps: -1
  max_time: null
  min_epochs: null
  min_steps: null
  move_metrics_to_cpu: false
  multiple_trainloader_mode: max_size_cycle
  num_nodes: 1
  num_processes: null
  num_sanity_val_steps: 0
  overfit_batches: 0.0
  plugins: null
  precision: 32
  profiler: null
  reload_dataloaders_every_n_epochs: 0
  replace_sampler_ddp: true
  strategy: null
  sync_batchnorm: false
  tpu_cores: null
  track_grad_norm: -1
  val_check_interval: 1.0 # Don't validate before extracting features.
```

**Descriptions**

- PyTorch-Lightning からの`trainer`構成ファイル内のすべての属性のリスト. 詳細は [こちら](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)をご覧ください。