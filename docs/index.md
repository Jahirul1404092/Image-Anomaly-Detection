# Hamacho

A **CUI** product to perform visual anomaly detection. Developed to streamline PoC. By preparing a non-defective image and an abnormal image, you can output images for analysis reports. 

- Reference PPT File - [Hamacho Code Overview](https://chowagiken.sharepoint.com/:p:/g/CorporatePlanning/licence-business/EZEn3r3DjkhMrDrRyVq4BvMBshJtDOGh9jR2hKYBQ8Z73g?e=oSiaFF). This is the original document in PPT format. 


## GitHub 

In github read-me page, there are also some document that can be useful. Placing it here for reference.

- [Prerequisites](https://github.com/chowagiken/anomaly_detection_core_hamacho#Prerequisites%E5%89%8D%E6%8F%90%E6%9D%A1%E4%BB%B6)
- [Environment-Set-up](https://github.com/chowagiken/anomaly_detection_core_hamacho#Environment-Set-up%E7%92%B0%E5%A2%83%E6%A7%8B%E7%AF%89)
- [Data-Preparation](https://github.com/chowagiken/anomaly_detection_core_hamacho#Data-Preparation%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E6%BA%96%E5%82%99)
- [Training](https://github.com/chowagiken/anomaly_detection_core_hamacho#Training%E8%A8%93%E7%B7%B4)
- [Inference](https://github.com/chowagiken/anomaly_detection_core_hamacho#Inference%E6%8E%A8%E8%AB%96)
- [Argument-list](https://github.com/chowagiken/anomaly_detection_core_hamacho#Argument-list%E5%BC%95%E6%95%B0%E4%B8%80%E8%A6%A7)


## Documentation
- [QuickStart](./quickstart/quickstart.md)
- [API](./api/manage-api.md)
- [Config](./config/manage-config.md)
- [Docker](./docker/manage-docker.md)
- [Docker Compose WebUI Easy Startup](./docker/manage-docker-compose-webui.md)
- [Dataset](./dataset/manage-dataset.md)
- [Models](./model/manage-models.md)
- [Training](./training/manage-training.md)
- [Inference](./inference/manage-inference.md)
- [general](./general.md)

---

## 📘 High-level Operations 

Some of the code logic or operations can be placed under below with details. 

### 📒 Number-of-Workers
`--num-workers`: It is one of a training argument in `hamacho` product. But currently it is not available for Windows system because there are some PyTorch issue. But it is available for Linux system because that issue doesn't occur there.​ So, in our code, if user tries to use it, then at first the code checks the operating system (OS). If the OS is Linux, then –num_workers will be used. But if it is Windows, it will prompt to user that, it is not available.​

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

### 📒 Inference Saving Path

`--save-path`: It is another CUI argument for the saving path location to save inference test results.​ If `–save-path` is `None` or not given, then the code will programmatically create two new directory called `inference` and `project_path`.

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

If `–save-path` is not `None` or given, then, the above path directory will be updated accordingly with the given.​

```python
# hamacho/main.py
if save_path is not None:
    Path(save_path).mkdir(exist_ok=True, parents=True)
    config.project.save_root = save_path
    config.project.infer_dir_name = ""
```

### 📒 Adaptive Threshold

`Adaptive Threshold`: Adaptive threshold is calculated at the training time.​ It is calculated by finding the maximum value of `F1-score`.

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

The adaptive threshold hold is calculated for image-level and also pixel-level (if mask images are provided in training time).


```python
# hamacho/plug_in/models/components/base/anomaly_module.py 
self.image_threshold = AdaptiveThreshold().cpu()
self.pixel_threshold = AdaptiveThreshold().cpu()
```

This adaptive value can be any number and that depends on the output of the model.​ This adaptive value will be later used to normalize the anomaly scores. And also, this threshold value will be shifted to `0.5` for better anomaly detection decision.​


### 📒 Min-Max Normalization​

The anomaly output map is normalized with the raw anomaly map scores, adaptive threshold and min-max value that found during training time.​

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

In the above code, 

- `target`: raw-anomaly-score​
- `threshold` : adaptive threshold value (image-level and pixel-level)​
- `max_val and min_val`: got from training set during training​
- `+0.5`: shifted the overall scores towards `0.5`.​
- `normalized`: anomaly score values `[0...1]` values.​

If normalized value greater than `0.5` is considered​
as `NG` and below than that is considered as `OK`.​ However, in inference time, this threshold value ​(`0.5`), can be changed if we want.

### 📒 Customizing Threshold

In **inference** time, we can control the threshold parameter. See below, Code Link 1.​

```yaml
# hamacho/plug-in/model/config.yaml
threshold:
    image_norm: 0.5 # must be within 0 and 1
    pixel_norm: 0.5 # must be within 0 and 1
    image_default: 0
    pixel_default: 0
```

Here is the logic of how threshold value would be set in inference time.

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


​

​

​

​