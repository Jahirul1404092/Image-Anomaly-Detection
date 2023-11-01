# Model Serving

## Using TorchServe
TorchServe is a flexible and easy to use tool for serving and scaling PyTorch models in production.

Requires python >= 3.8

#### Architecture

![arch](https://user-images.githubusercontent.com/880376/83180095-c44cc600-a0d7-11ea-97c1-23abb4cdbe4d.jpg)

### Installation
If hamacho is already installed in your environment:
- Install openjdk 17 (Ubuntu)
```bash
sudo apt install openjdk-17-jdk
```
- Install torchserve with pip
```bash
pip install torch-model-archiver torchserve
```

### Generate MAR file
```bash
torch-model-archiver \
    --model-name <category-name> \
    --version <model-version> \
    --serialized-file <path to trained_data.hmc> \
    --handler <path to torchserve_handler.py> \
    --extra-files <path to generated config.yaml file upon training> \
    --export-path <path to model_store directory>
```

### Serve Model
Currently there are two ways to serve a model

#### 1. From command line
This command will initialize the model with default properties (min_workers=1, max_workers=1, batch_size=1, max_batch_delay=50).
```bash
torchserve --model-store ./model_store/ --models <model-name>=<model-mar-filename>.mar>
```

#### 2. From command line with config.properties
This command will initialize the models specified in the config file
```bash
torchserve --model-store ./model_store/ --ts-config config.properties
```
Sample config.propertis file contents:
```
load_models=<model-mar-filename>.mar,<model-mar-filename>.mar
models={\
  "<model-name>": {\
    "0.1": {\
        "defaultVersion": true,\
        "marName": "<model-mar-filename>.mar",\
        "minWorkers": 2,\
        "maxWorkers": 4,\
        "batchSize": 5,\
        "maxBatchDelay": 10,\
        "responseTimeout": 120\
    }\
  }\,
  ...
}
```

#### 3. From Management API
To use the Management API, TorchServe must be started:
```
torchserve --model-store <path to model store directory>
```

After generating a mar file place it in the model store directory and start the model. Detailed info is [here](https://pytorch.org/serve/management_api.html#register-a-model). Example:
```
curl -X POST "http://localhost:8081/models?url=<model-mar-filename>.mar&batch_size=5&initial_workers=2&max_batch_delay=15"
```


### Scale Workers
To scale the workers, use the Management API. Currently only `min_worker`, `max_worker` and `synchronous` parameters are allowed in API. For more detail, see [here](https://pytorch.org/serve/management_api.html#scale-workers). Example:
```
curl -X PUT "http://localhost:8081/models/<model-name>?min_worker=1&synchronous=true"
```

**CAUTION:** if any wrong parameter is used on this API, the workers will get scaled down to 1. So, please pay extra caution and check the spelling of the allowed parameters.


### Inference
To do inference on 1 image:
```
curl http://127.0.0.1:8080/predictions/<model-name> -T <path to image>
```

For multiple image:
```
curl http://127.0.0.1:8080/predictions/<model-name> -T "{<path to image1>,<path to image2>,...}"
```

Inference on a specific model version is supported:
```
curl http://127.0.0.1:8080/predictions/<model-name>/<version> -T <path to image>
```
