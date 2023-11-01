# How to use CLI
   ![動作イメージ](../assets/hamacho_cli.gif)

*********************

# environment building

## How to install with pip

The whl file can be found [here](https://chowagiken.sharepoint.com/CorporatePlanning/licence-business/Shared%20Documents/Forms/AllItems.aspx?newTargetListUrl=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents&viewpath=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2FForms%2FAllItems%2Easpx&id=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2F02%5F%E7%95%B0%E5%B8%B8%E6%A4%9C%E7%9F%A5%5FAnomalyDetection%2F04%5F%E3%83%97%E3%83%AD%E3%83%80%E3%82%AF%E3%83%88%E9%96%8B%E7%99%BA%5FDevelopment%2F03%5F%E9%96%8B%E7%99%BA%E6%B8%88%E3%81%BF%E6%8F%90%E4%BE%9B%E3%83%A9%E3%82%A4%E3%83%96%E3%83%A9%E3%83%AA%2Fanomaly%5Fdetection%5Fcore%5Fhamacho%2Fv1%2E4%2E0&viewid=d90a7d83%2Dbcee%2D4d76%2D8485%2D0947a8795bdc).

How to Install example
```sh
pip install dist/hamacho-1.4.0-py3-none-any.whl
```

# Data Preparation

- Download the sample dataset (bottel-sealing-surface) [here](https://chowagiken.sharepoint.com/CorporatePlanning/licence-business/Shared%20Documents/Forms/AllItems.aspx?newTargetListUrl=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents&viewpath=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2FForms%2FAllItems%2Easpx&id=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2F02%5F%E7%95%B0%E5%B8%B8%E6%A4%9C%E7%9F%A5%5FAnomalyDetection%2F04%5F%E3%83%97%E3%83%AD%E3%83%80%E3%82%AF%E3%83%88%E9%96%8B%E7%99%BA%5FDevelopment%2F01%5F%E8%A9%95%E4%BE%A1%E7%94%A8%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%2F%E5%95%86%E7%94%A8NG%EF%BC%89mvtec%5Fbottle&viewid=d90a7d83%2Dbcee%2D4d76%2D8485%2D0947a8795bdc)  
  (MVTec's bottle data is thinned out)

Prepare the folders so that they have the following structure.

```
root
│ 
├── data
│   ├──<category1>
│   │   ├── good
│   │   │    └─ [SOME GOOD IMAGE]
│   │   ├── bad
│   │   │    └─ [SOME BAD IMAGE]
│   │   └── mask [optional]
│   │        └─ [SOME MASK LABEL IMAGE W.R.T BAD IMAGE]
│   │ 
│   ├──<category2>
│   │   ├── good
│   │   ├── bad
│   │   └── mask [optional]
```

# Train
Training is performed in the following steps  

1. generate an anomaly detection model using the normal images in data-category-good.  
At this time, the images in good are automatically split (training:evaluation = 8:2).  
2. evaluation is also performed within this command using the generated model.  
The evaluation is performed using 20% of the good images and all of the bad images.   
3. The training model and evaluation result images will be output to the results folder below.    

****
There are three options for learning   
- [Segmentation (with mask)](#1-option-1)  
- [Segmentation (without mask)](#2-option-2)  
- [Classification](#3-option-3)  
We also offer [no-bad-mode](#special-condition-anomalous-samples-unavailableno-bad-mode), which can be trained using only normal images.

****

### Output after Training  

```
root
│ 
├── data(data of images)
│   └──<category1>
│       ├── good
│       │   └─ [SOME GOOD IMAGE]
│       ├── bad
│       │   └─ [SOME BAD IMAGE]
│       └── mask [optional]
│           └─ [SOME MASK LABEL IMAGE W.R.T BAD IMAGE]
└──results(outputs)
    └──<category>
        └──<model name>
            ├── test_predictions
            │   │
            │   ├── images
            │   │   ├── combined(histgram and result images)
            │   │   │    └── [SOME RESULT IMAGE]
            │   │   ├── grayscale(Image score per pixel)
            │   │   │    └── [SOME RESULT IMAGE]
            │   │   ├── predicted_heat_map(Show anomaly)
            │   │   │    └── [SOME RESULT IMAGE]
            │   │   └──predicted_mask(Mask GrayScale image)
            │   │        └── [SOME RESULT IMAGE]
            │   │       
            │   ├── csv(Image score text per pixel)
            │   │        └── [SOME RESULT csv file]
            │   │
            │   └── metrics(AUROC,confusion_matrix,image-level-roc,etc)
            │            └── [SOME RESULT IMAGE and csv file]
            │           
            ├── weights
            │    └── trained_data.hmc
            │          
            └── config.yaml
```
The pixel Level output is displayed only when the with-mask-label (default False) option is enabled.

|     |  Test metric  | DataLoader 0       |
| --- | :-----------: | :------------------: |
||     Image Level AUROC       |          1.0|
||Image Level BinaryAccuracy   |  0.9848484992980957|
||Image Level BinaryPrecision  |          1.0|
|| Image Level BinaryRecall    |  0.9841269850730896|
||    Image Level F1Score      |          1.0|
||     Pixel Level AUROC       |  0.9528321623802185|
||Pixel Level BinaryAccuracy   |  0.9129900336265564|
||Pixel Level BinaryPrecision  |  0.4540739953517914|
|| Pixel Level BinaryRecall    |  0.9793243408203125|
||    Pixel Level F1Score      |  0.6204635500907898|

************


### 1. Option (1)
With mask labels on the data, the segmentation results after training & evaluation will be as follows

─────────────────────────────
### For training with PIP installation

```sh
hamacho train --with-mask-label --task-type segmentation --model patchcore --category bottle-sealing-surface
```
Output result image (AUROC and other graphs on the right side are output in a separate file)
![image](https://user-images.githubusercontent.com/110383823/203926830-5fd6241d-d749-4fbb-80cb-18c8ac49957a.png)


### 2. Option (2)
The segmentation results after training & testing, with no mask labels in the data, are as follows

```sh
hamacho train  --task-type segmentation --model patchcore --category bottle-sealing-surface
```

Output result image (AUROC and other graphs on the right side are output in a separate file)
![image](https://user-images.githubusercontent.com/110383823/203928874-9772720b-c896-4631-b963-c03c5ecb6f23.png)

### 3. Option (3)
The results of the classification after training & testing with no mask labels in the data are as follows

```sh
hamacho train  --task-type classification --model patchcore --category bottle-sealing-surface
```

Output result image (AUROC and other graphs on the right side are output in a separate file)
![image](https://user-images.githubusercontent.com/110383823/203927906-4af2dbaf-bbd5-4160-9575-9aecd357bf00.png)

### Special Condition: Anomalous Samples Unavailable(no-bad-mode)
 By adding the `--no-bad-mode argument`, it is possible to train [segmentation (no mask)](#2-option-2) and [classification](#3-option-3) (good) on normal images only.

- Prepare the following data structures:
  ```
  data
  └── <product category>
      │
      └── good
          └── [SOME GOOD IMAGE]
  ```
- train:
  
  ```sh
  hamacho train --task-type [segmentation/classification] --model [patchcore/padim] --category bottle-with-no-bad --no-bad-mode
  ```

# inference
*****
Inference is performed by inputting an arbitrary image and outputting the percentage of anomaly for that image (100% is anomaly) and the ImageScore of the inferred image.

*****

```
root
│ 
├── data(data of images)
│   └──<category>
│       ├── good
│       │   ├── [SOME GOOD IMAGE]
│       ├── bad
│       │   ├── [SOME BAD IMAGE]
│       └── mask [optional]
│           └── [SOME MASK LABEL IMAGE W.R.T BAD IMAGE]
└── results(outputs)
   └──<category>
       └──<model name>
           └── inference
               │
               ├── images
               │   ├── combined(histgram and result images)
               │   │        └── image-path name
               │   │             └── [SOME RESULT IMAGE]
               │   │   
               │   ├── grayscale(Image score per pixel)
               │   │        └── image-path name
               │   │             └── [SOME RESULT IMAGE]
               │   │   
               │   ├── predicted_heat_map(Show anomaly)
               │   │        └── image-path name
               │   │             └── [SOME RESULT IMAGE]
               │   │   
               │   └── predicted_mask(Mask GrayScale image)
               │            └── image-path name
               │                 └── [SOME RESULT IMAGE]
               │       
               ├── csv(Image score text per pixel)
               │       └── [SOME RESULT csv file]
               │
               └── metrics(Predict result,Predict file name...etc )
                       └── test_outputs.csv
```

There are two options for inference.  
- [Infer one file](#when-inferring-a-single-piece-of-data)
- [Infer all images in a folder](#when-inferring-all-images-in-a-folder)

## When inferring a single piece of data

```sh
hamacho inference --image-path "data/bottle-sealing-surface/000.png" \
                         --config-path "./results/bottle-sealing-surface/patchcore/config.yaml" \
                         --save-path "./results/infer"
```

## When inferring all images in a folder

```sh
hamacho inference --image-path "data/bottle-sealing-surface/bad" --config-path "results/bottle-sealing-surface/patchcore/config.yaml"
```

result：
Single or multiple outputs are the same (output varies by task type)

Segmentation output results
![image](https://user-images.githubusercontent.com/110383823/203935693-9774d2ed-45f4-4dbb-bd3e-ef32e660a0c9.png)

Classification output results　（ImageScore<0.5 is normaly）
![image](https://user-images.githubusercontent.com/110383823/203933763-6e63b23b-ca14-4ab1-84e8-26b5ea7119c6.png)


## Control Threshold
The default value of the image threshold (image_norm) is 0.5 or less for normal and 0.5 or more for abnormal.

In the case of segmentation, the mask value can be changed by controlling the threshold value (pixel_norm) for each pixel.

```yaml
  threshold:                    threshold:
    image_norm: 0.5       →       image_norm: 0.5
    pixel_norm: 0.5       →       pixel_norm: 0.9
    image_default: 0      →       image_default: 0
    pixel_default: 0      →       pixel_default: 0
    adaptive: true        →       adaptive: true
```
![000](https://user-images.githubusercontent.com/110383823/204235121-02416dc2-bbb5-407c-abee-643fbfa8e013.png)

---


# Argument-list
## Argument-list for train  
Usage：Usage: main.py train [OPTIONS]  

| required | argument | type | mark                                                                                                                                                                                                             |
| ---------------- | :---------------: | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 〇               |    --category     | TEXT       | Name of the product category in the dataset root folder                                                                                                                                                                  |
| ×                |      --model      | TEXT       | Name of the algorithm to train/test<br>[default] patchcore<br>[select] patchcore or padim                                                                                                                                |
| ×                |   --result-path   | TEXT       | Path to save the results<br>[default]　./result                                                                                                                                                                          |
| ×                |    --task-type    | TEXT       | Whether modeling as classificaiton or segmentation approach.<br>[default] segmentation <br>[select] classificaiton or segmentation                                                                                       |
| ×                | --with-mask-label | BOOLEAN    | We can train model with or without ground truth mask. <br>[default] False                                                                                                                                                |
| ×                |   --accelerator   | TEXT       | You can select cpu or gpu or auto for the device.<br>[default] auto                                                                                                                                                      |
| ×                |  --dataset-root   | PATH       | Directory of the dataset root folder  <br>[default] ./data                                                                                                                                                               |
| ×                |   --data-format   | TEXT       | Name of dataset format of the <product category> in the dataset root directory  <br>[default] folder  <br>[select] folder or mvtec                                                                                       |
| ×                |   --image-size    | INTEGER    | Image size of the training images in pixels. The value must be divisible by 32 and within 64 and 640. <br> [default] None (means the training will be done in the default image size value set in the model config file) |
| ×                |   --batch-size    | INTEGER    | Batch size of train and test dataloader. The value must be greater than 0. <br> [default] None (means the batch size value set in the model config file will be used)                                                    |
| ×                |   --num-workers   | INTEGER    | Number of cpu processes that the data loader can use. Linux,MacOS only use <br>[default] 0                                                                                                                                                    |
| ×                |      --split      | FLOAT      | Percentage of train set images that will be used for test set. Must be within 0.01 and 0.99. <br>[default] 0.2                                                                                                           |
| ×                |      --seed       | INTEGER    | Seed value to control dataset random split <br>[default] 420                                                                                                           |
| ×                |   --config-path   | PATH       | [Expert Only] Edited config path of model config.yaml file <br>[default] None (means model config will be loaded automatically from --model arg)                                                                                                           |
| ×                |   --no-bad-mode   | BOOLEAN       | Select this when only good data is available <br>[default] False                                                                                                           |


## Argument-list for inference

Usage：Usage: main.py inference [OPTIONS]

| required | argument | type | mark                                                                                  |
| ---------------- | :---------------: | ---------- | --------------------------------------------------------------------------------------------- |
| 〇               |   --image-path    | PATH       | Path to image(s) to infer.[required]                                                          |
| 〇               |   --config-path   | TEXT       | Path to a config file  [required]<br>[default] results/<product_category>/<model>/config.yaml |
| ×                |    --save-path    | TEXT       | Path to save the output image(s).<br>[default] inference                                      |
| ×                |  --batch-size   | INTEGER    | Inference batch size.                                 |
| ×                |   --accelerator   | TEXT       | You can select cpu or gpu or auto for the device.<br>[default] auto                           |


## Unit/integration testing

To run all test cases:

```bash
{projecy-root}: pytest tests/ --disable-pytest-warnings
# or
{projecy-root}: python -m pytest tests/ --disable-pytest-warnings
```

To test individual functions:

```bash
{project-root}: pytest <path-to-py-file> --disable-pytest-warnings
                # example
                # pytest tests/integration/train/test_train_model.py --disable-pytest-warnings
```

#### Generating Test Coverage Reports
To generate a Test Coverage Report:

```bash
{project-root}: pytest --cov-report=html:coverage-report --cov=hamacho tests/ --disable-pytest-warnings
```
The above command will generate a folder named coverage-report. Open the coverage report.


[[back]](quickstart.md)