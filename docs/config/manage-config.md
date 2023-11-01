# Configuration File

Here we will know about the various attributes of the configuration file. The configuration file generally is placed under `hamacho/plug_in/{model}/config.yaml` locations. But this can also be passed from CUI. This `yaml` files contains the following high-level attributes, 

- `dataset`: It provides many sub-attributes for data-oriented configurable items. 
- `model`: It provides many sub-attributes for model related configurable items.
- `metrics`: It contians the collections of metrics that will be used for training, evaluation and inference time.
- `project`: Mostly contains attributes to control output style for evaluation and inference; boolean attributes to control logging etc.
- `trainer`: The `hamacho` code is heavily based on `PyTorch-Lightning`. This `trainer` class (and its attributes) are mostly from the core API.

---

Let's discuss each of this high-level attributes with details. We will discuss the most prominent attributes and skip the obvious one. However, please note, based on the type of model (i.e. `padim`, `patchcore`, etc), some of the model-oriented sub-attributes can be different.


## Data

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

**Descriptions**

- `name`: It can be any raw string that may represent the data format type.
- `format`: It defines the supported data-loader.  Currently, the supported data-loaders are `mvtec`, `folder` and `filelist` format.  
  - The `mvtec` format is for bechmarking **MVTec** dataset for research purpose. If you add new model, check it with these dataset. 
  - For custom data set, `folder` or `filelist` format can be set.
- `path`: The root directory of data set. For `mvtec` and `folder` format dataset, this is essentail.
- `task` : Supported option: `classification` and `segmentation`.​ 
    - If `classification` is set, then the model will be training with good and bad images. The resultant metric will be only image-level.
    - If `segmentation` is set, then the model will be training with good and bad images. The resultant metric will be only image-level. But the visual prediction may contains pixel-level predictions. Also, if `mask` samples are provided, then the resultant metric will be pixel-level additionally.
- `category`: Generally, it is a folder name, placed inside the `path` directory. It will represent as product id or name etc. For example, if you need to find anomaly from carpet object, then you should name your `category` as `carpet`.
- `num_workers`: Number of CPU workers, mostly efficient for data processing.
- `transform_config`: By this, it is possible to do image augmentation. Useful it the model is trained with multiple epoch. ​
- `normal_dir / abnormal_dir / normal_test_dir`: [Check](https://chowagiken.slack.com/archives/C031VDC258T/p1668662477809799).​

- `tiling`: It may be useful for bigger input image training. By setting it to `True`, the image will be extracted into multiple patches for efficient training. For example, let's say your target anomaly object is too small compared to the image size, then `tiling=True` can be useful, as each extracted patch will be treated separately and the small anomaly will have most possiblity to be captured.
    - `apply`: Boolean flag, `true` or `false` whether to use it or not. 
    - `tile_size`: Tile size. Size of the extracted patches.
    - `stride`: Stride to move tiles on the image.
    - `remove_border_count`: Number of pixels to remove from the image before tiling.
    - `use_random_tiling`: Boolean flag, `true` or `false`.
    - `random_tile_count`: Number of random tiles to sample from the image.
  
    In the literature on deep learning, input image sizes typically range from `224` to `768` pixels. In the majority of industrial applications, however, input image sizes are significantly larger. Before the forward pass, these images are resized to a smaller scale before being fed into the models. However, this is problematic in the context of anomaly detection tasks, where the anomaly is typically **quite small**. The detection of abnormalities becomes extremely difficult when the image is shrunk in size. A common method for addressing this issue is to tile the input images so that no information is lost during the resizing operation. [Details.](https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/100_datamodules/104_tiling.ipynb)

---

## Model

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
  normalization_method: min_max # options: [null, min_max, cdf, sigma6]
```

**Descriptions**

- `name`: Name of the supported model. Currently, it is `patchcore` and `padim`. Patchcore is better than padim in most cases. But patch-core is a bit memory-inefficient. 
- `backbone` : The backbone model. It is used to extract the feature representation of the samples. Both model use different backbone. In patch-core, it is `wide_resnet50_2` and in padim, it is `resnet18`, by default. These backbone can be changed. But note, you may need to do some test to find better options for the following attributes.
- `layers`: Number of the feature output vectors from backbone model. All these output vectors are merged together to build up the rich feature representation.​ In padim (backbone: `resnet18`), the layers are `layers1, layers2, layers3`. For patchcore (`wide_resnet50_2`), the layers are `layer2, layer3`. FYI, the general idea to choose layer is to take from around low to mid-level of the backbone. High level layers are more inclined to ImageNet dataset. 
- `normalization_method`: It is a normalization method, typically applied on the output of raw anomaly map scores that is received from the model. Supported options are min-max, cdf (cumulative density function), sigma6. 
    - `min-max`: Min-max normalization method applied on the anomaly scores.
    - `cdf`: Cumulative distribution function (or **CDF**). One of the normalization method, standardizes the image-level and pixel-level anomaly scores. DOI: 10.1109/TGRS.2014.2387574.
    - `sigma6`:​ This normalization method is applicable when the model is trained with only good samples (without any bad samples). Specially, when `--no-bad-mode` is enabled in the CUI. 
- `coreset_sampling_ration`: Coreset sampling ratio to subsample embedding. Relate to `patch-core` model.​
- `num_neighbors`: Number of  nearest neighbors. ​Relate to `patch-core` model only.

---

## Metrics

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
    adaptive: true
```

**Descriptions**

- `image`: List of metrics, which will be used to evaluate the model performance. 
- `pixel` : List of metrics, which will be used to evaluate the model performance. To get results from this pixel-level metrics, one should provide binary mask data set as a ground truth of bad images.​
- `threshold.adaptive`: When it is True, threshold value will be evaluated auto. It will be done by maximizing the `F-1` score.
- `threshold.image_norm`: At inference time, this attritube can be used to tweak the threshold values (range form `0-1`).​
- `threshold.pixel_norm`: same as `image_norm`.​


---

## Project

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

**Descriptions**

- `path`: It's saving path location as root directory. 
- `log_images_to`: See the external documentation. [LINK​](https://openvinotoolkit.github.io/anomalib/tutorials/logging.html)
- `save_outputs.image.segmentation`: It contains the list of outputs that a user wants to get after running the evaluation on a dataset. Basically, for `task=segmentation`, these outputs will be saved in local disk.
- `save_outputs.image.classification`: Similarly for `task=classification` cases.
- `save_outputs.image.csv`: The anomaly map (scores) will be saved in `CSV` format, if `True`.
- `save_combined_result_as_image`: If `True`, the output of `save_outputs.image.{task-type}` will be saved together.
- `add_label_on_image`: If `True`, the predicted label will be placed on top of the input image as text. This flag is only valid for `Classification` results

---

## Trainer

```yaml
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

- All the list of `trainer` attributes in config file from PyTorch-Lightning. See details [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).