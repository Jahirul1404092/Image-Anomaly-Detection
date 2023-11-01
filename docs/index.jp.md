# Hamacho

異常検知コア (hamacho)

異常検知のCUIプロダクトです。社内のPoCを効率化するために開発されました。 良品画像・異常画像を準備すれば、分析レポート用の画像を出力することができます。

- 参照 - [Hamacho Code Overview](https://chowagiken.sharepoint.com/:p:/g/CorporatePlanning/licence-business/EZEn3r3DjkhMrDrRyVq4BvMBshJtDOGh9jR2hKYBQ8Z73g?e=oSiaFF). PowerPointで書かれたドキュメント。 



## GitHub 

githubのREADMEページには、参考になるドキュメントもあります。参考までにここに載せておきます。

- [提供形式](https://github.com/chowagiken/anomaly_detection_core_hamacho/blob/WAD-364-Organizing-documents/README.md#%E6%8F%90%E4%BE%9B%E5%BD%A2%E5%BC%8F)
- [環境設定](https://github.com/chowagiken/anomaly_detection_core_hamacho/blob/WAD-364-Organizing-documents/README.md#%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E6%BA%96%E5%82%99)
- [データ準備](https://github.com/chowagiken/anomaly_detection_core_hamacho/blob/WAD-364-Organizing-documents/README.md#%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E6%BA%96%E5%82%99)
- [トレーニング](https://github.com/chowagiken/anomaly_detection_core_hamacho/blob/WAD-364-Organizing-documents/README.md#train%E5%AD%A6%E7%BF%92)
- [Inference](https://github.com/chowagiken/anomaly_detection_core_hamacho/blob/WAD-364-Organizing-documents/README.md#inference%E6%8E%A8%E8%AB%96)


## Documentation
- [Quick Start Guide](./quickstart/quickstart.jp.md)
- [API](./api/manage-api.jp.md)
- [設定](./config/manage-config.jp.md)
- [Docker](./docker/manage-docker.md)
- [Docker Compose WebUI Easy Startup](./docker/manage-docker-compose-webui.jp.md)
- [データセット](./dataset/manage-dataset.jp.md)
- [モデル](./model/manage-models.jp.md)
- [トレーニング](./training/manage-training.jp.md)
- [推論](./inference/manage-inference.jp.md)
- [general](./general.jp.md)



---

## 📘 高レベルな操作 

性能を引き出したり、効率化するための操作方法とその仕組みを下記に記載いたします。
  
  
### 📒 Number-of-Workers（CPUの並列化）

`--num-workers`:  
データセットを読み込む際、複数のCPUを割り当て、データ読込みを高速化することができる引数です。  
現在、PyTorchの問題(バグ)のため、Windowsでは使用することができません。（メモリ割り当てに失敗する）  
Linuxではその問題は発生しないためこの引数を利用することができます。  
  
以下のコードは、この機能の詳細を説明しています。  

 - ユーザーがこの引数を使用しようとすると、まずオペレーティング システム (OS) をチェックします。
 - OS が Linux の場合、–num_workers が使用されます。
 - Windows の場合は、利用できないことをユーザーに確認するメッセージが表示されます。

```python
# hamacho/main.py
hide_nw = False
if platform.system().lower() != "linux":
    hide_nw = True
...
@click.option(
    "--num-workers",
    type=NumWorkersType(), default=0, hidden=hide_nw,
    help="Number of workers for the dataloader to use"
)

...

# hamacho/core/utils/click_types.py 
class NumWorkersType(click.ParamType):
    current_system = platform.system().lower()
    supported_system = "linux"
    name = "integer"
    default, lower_limit = 0, 0
    cpu_headroom = 2
    cpu_count = get_cpu_count()
    upper_limit = cpu_count - cpu_headroom

    def convert(self, value: str, param, ctx) -> int:
        if value != self.default and self.current_system != self.supported_system:
            raise click.NoSuchOption("--num-workers", f"No such option: --num-workers")
```

### 📒 推論の保存先

`--save-path`: 推論のテスト結果を保存するための「保存パスを指定」する引数です。  

 - もし `-–save-path` が `None` か、指定されていない場合
     - `inference`と `project_path`(e.g:bottle-sealing-surface)という名前の 2 つの新しいディレクトリをプログラムで自動的に作成します。
以下のコードは、この機能の詳細を説明しています。

```python
# hamacho/main.py
@click.option(
    "--save-path",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="Path to save the output image(s)."
)

# hamacho/core/config/config.py
if "infer_dir_name" not in config.project.keys():
    config.project.infer_dir_name = "inference"

(project_path / "weights").mkdir(parents=True, exist_ok=True)
(project_path / config.project.test_dir_name).mkdir(parents=True, exist_ok=True)
config.project.path = str(project_path)
config.project.save_root = str(project_path)
```

 `–save-path` が `None` でない場合、もしくは指定されている場合  

 - 上記のパスディレクトリは、指定された内容に応じて更新されます。​

```python
# hamacho/main.py
if save_path is not None:
    Path(save_path).mkdir(exist_ok=True, parents=True)
    config.project.save_root = save_path
    config.project.infer_dir_name = ""
```

### 📒 Adaptive Threshold(適応閾値)

`Adaptive Threshold`:   

- 学習時にて適応閾値※（Adaptive threshold）を計算します。
- `F1-score`の最大値を求めて計算します。
- ※適応閾値とは、閾値を固定せず、値によって変化させる二値化処理です。
- ※この閾値以上の場合、異常と判断します。

```python

precision, recall, thresholds = self.precision_recall_curve.compute()
f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
if thresholds.dim() == 0:
    # special case where recall is 1.0 even for the highest threshold.
    # In this case 'thresholds' will be scalar.
    self.value = thresholds
else:
    self.value = thresholds[torch.argmax(f1_score)]
return self.value
```

- 適応閾値はイメージレベル（画像全体の異常スコア）とピクセルレベル（画素単位の異常スコア）それぞれ計算されます。
- ピクセルレベルに関しては、学習時にマスク画像が提供された場合のみ計算されます。


```python
# hamacho/plug_in/models/components/base/anomaly_module.py 
self.image_threshold = AdaptiveThreshold().cpu()
self.pixel_threshold = AdaptiveThreshold().cpu()
```

- この適応閾値は任意の数値にすることができます。閾値の値はモデルに依存します。
- この適応閾値は、後で異常スコアを正規化するために使用されます。
- また、この閾値をシフトすることによって、異常判定を改善することができます。


### 📒 Min-Max Normalization​

異常出力マップは、モデルから出力された生の異常スコア、適応閾値、学習時に算出した異常スコアの最大、最小値を使って  
0.0-1.0の範囲で規格化されます。  

以下のコードは、この機能の詳細を説明しています。

```python
#hamacho/core/post_processing/normalization/min_max.py
def normalize(
    targets: Union[np.ndarray, Tensor, np.float32],
    threshold: Union[np.ndarray, Tensor, float],
    min_val: Union[np.ndarray, Tensor, float],
    max_val: Union[np.ndarray, Tensor, float],
) -> Union[np.ndarray, Tensor]:
    """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
    normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    if isinstance(targets, (np.ndarray, np.float32)):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, Tensor):
        normalized = torch.minimum(
            normalized, torch.tensor(1)
        )  # pylint: disable=not-callable
        normalized = torch.maximum(
            normalized, torch.tensor(0)
        )  # pylint: disable=not-callable
    else:
        raise ValueError(
            f"Targets must be either Tensor or Numpy array. Received {type(targets)}"
        )
    return normalized
```

上記のコードでは、 

- `target`: 生の異常スコア​
- `threshold` : 適応閾値 (画像レベルおよびピクセルレベル)​
- `max_val and min_val`: 学習中に学習データの中から取得
- `+0.5`: 全体的なスコアを`0.5`.にシフトした。（本来基準は0.0から±0.5になる）
- `normalized`: 正規化された異常スコアの値（範囲は `[0...1]` ​）

正規化された値が`0.5`より大きい場合は`異常`とみなされ、それより小さい場合は`正常`とみなします。ただし、推論時にこの閾値​(`0.5`)は、必要に応じて変更することができます。

### 📒 閾値のカスタマイズ

推論時（**inference**）に閾値パラメータを制御することができます。   

以下のコードを参照してください。

```yaml
# hamacho/plug-in/model/config.yaml
threshold:
    image_norm: 0.5 # must be within 0 and 1
    pixel_norm: 0.5 # must be within 0 and 1
    image_default: 0
    pixel_default: 0
```
これは、推論時に閾値がどのように設定されるかの理論です。

```python
# hamacho/core/utils/callbacks/min_max_normalization.py
def _set_thresholds(self, pl_module: AnomalyModule) -> None:
    if pl_module.image_metrics is not None \
        and self.norm_image_threshold is not None:
        pl_module.image_metrics.set_threshold(self.norm_image_threshold)
    elif pl_module.image_metrics is not None:
        pl_module.image_metrics.set_threshold(0.5)

    if pl_module.pixel_metrics is not None \
        and self.norm_pixel_threshold is not None:
        pl_module.pixel_metrics.set_threshold(self.norm_pixel_threshold)
    elif pl_module.pixel_metrics is not None:
        pl_module.pixel_metrics.set_threshold(0.5)
```
