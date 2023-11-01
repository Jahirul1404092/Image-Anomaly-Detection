# How to use API

## contents 
- [How to buld docker image](#deployment-of-production-environment)
- [execute container](#execute-container)
- [How to use API from windows](#how-to-use-api-from-windows-use-wsl2)


Prepare a data set with the following structure

```
C:
│ 
└── visee-anodet
    ├── data
    │   └──bottel
    │       ├── good
    │       │    └── [SOME GOOD IMAGE]
    │       ├── bad
    │       │    └── [SOME BAD IMAGE]
    │       └── mask [optional]
    │            └── [SOME MASK LABEL IMAGE W.R.T BAD IMAGE]
    └── license
         └── license.lic
```

## Deployment of production environment

### Shared library and container image creation with Cython
(Execute the following command in the repository root)
```
./docker/build_image.sh     # <-- select prod-api-obfuscated
```

### For development and test environments
```
./docker/build_image.sh     # <-- select api or api-test
```

## execute container
The path setting of ***HOST_DATA_DIR*** to use the API differs depending on the OS.
running API with `hamacho:v1.4.0-api-test` or `hamacho:v1.4.0-api` image requires more arguments.
License file mount:
#### How to use API from windows (use WSL2)

```
docker run --name hamacho --gpus all \
-e HOST_DATA_DIR=C:\\\\visee-anodet\\\\data \
-e BASE_DATA_DIR=/data \ 
-e DBPATH=/app/hamacho_db.sqlite \
-e PREDICTION_MODE=SYNC \
-e PORT=5000 \
-v /mnt/c/visee-anodet/data:/data \
-v /mnt/c/visee-anodet/license:/license \
-v /mnt/c/visee-anodet/results:/app/results/ \
-p 5000:5000 \
hamacho:v1.4.0-prod-api-obfuscated
```

```
docker run --name hamacho --gpus all \
-e HOST_DATA_DIR=C:\\\\visee-anodet\\\\data \
-e BASE_DATA_DIR=/data \ 
-e DBPATH=/app/hamacho_db.sqlite \
-e PREDICTION_MODE=SYNC \
-e PORT=5000 \
-v /mnt/c/visee-anodet/data:/data \
-v /mnt/c/visee-anodet/license:/license \
-v /mnt/c/visee-anodet/results:/app/results/ \
-e OBFUSCATED_VERIFIER_DUMMY_DIR=/app/api/util \
-e PYARMOR_LICENSE=/app/license/license.lic \
-p 5000:5000 \
hamacho:v1.4.0-api
```

#### Example of using the API from a WSL (executed in a WSL)
```
docker run --name hamacho --gpus all \
-e HOST_DATA_DIR=/mnt/c/visee-anodet/data \
-e BASE_DATA_DIR=/data \
-e DBPATH=/app/hamacho_db.sqlite \
-e PREDICTION_MODE=SYNC  \
-e PORT=5000 \ 
-v /mnt/c/visee-anodet/data:/data \
-v /mnt/c/visee-anodet/license:/license \
-v /mnt/c/visee-anodet/results:/app/results/ \
-p 5000:5000 \ 
hamacho:v1.4.0-prod-api-obfuscated
```
#### Example of using Windows docker desktop
```
docker run --name hamacho --gpus all \
-e HOST_DATA_DIR=C:\\\\visee-anodet\\\\data \
-e BASE_DATA_DIR=/data \
-e DBPATH=/app/hamacho_db.sqlite \
-e PREDICTION_MODE=SYNC \
-e PORT=5000 \
-v C:\\Users\\visee-anodet\\data:/data \
-v C:\\visee-anodet\\license:/license \
-v C:\\visee-anodet\\results:/app/results/ \
-p 5000:5000 \
hamacho:v1.4.0-prod-api-obfuscated
```
#### How to start the container for the second and subsequent times with the container in place
```
docker start hamacho
```

## How to use the API in Python from Windows (use WSL2)
Run the API in Python from cmd as follows



```python
"""
Server startup
Execute the following in another terminal.
Assuming data is under data/bottle/.
$ python demorun.py
"""

import requests
from pathlib import Path
import json

host = "http://localhost:5000/"

good = []
for p in Path("data/bottle/good").glob("*.png"):
    good.append(str(p.absolute()))

bad = []
for p in Path("data/bottle/bad").glob("*.png"):
    bad.append(str(p.absolute()))

res = requests.post(
    f"{host}/addimage",
    json={"image_tag": "bottle", "image_path": good[:-2], "group": "good"},
    headers={"content-type": "application/json; charset=cp932"},
)
print("image_id:", res.json())

res = requests.get(f"{host}/listimages?tag=bottle")
print("images:", res.json())

res = requests.get(f"{host}/imagedetails?image_tag=bottle")
print("image details:", res.json())

res = requests.post(
    f"{host}/train",
    json={
        "image_tag": "bottle",
        "model_tag": "bottle",
    },
)
print("model_id:", res.json())

res = requests.get(f"{host}/listmodels")
print("models:", res.json())

res = requests.post(f"{host}/servemodel", json={"tag": "bottle"})
print("result:", res)

res = requests.post(
    f"{host}/predict",
    json={"tag": "bottle", "image_paths": good[-2:] + bad[-2:], "save": True},
)
print("result:", res.json())

```

### Model deletion
```sh
curl -X DELETE http://localhost:5000/delmodel -H "Content-Type: application/json" -d "[3]"
```

### Model Unregistration
```sh
 curl -X DELETE http://localhost:5000/unservemodel?model_id=1
```

### Delete Registered Image
```sh
curl -X DELETE http://localhost:5000/delimage -H "Content-Type: application/json" -d '{"image_tag":"bottle"}'
```




For more information, click [here](./api/manage-api.md).


[[back]](quickstart.md)