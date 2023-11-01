# Deploying to production

## Creating a shared library and creating a container image using Cython

```
(in the repository root)
. /docker/build_image.sh # <-- select prod-api-obfuscated
```

## For development/test environment
```
. /docker/build_image.sh # <-- select api or api-test
```

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



## Container start
Example of using the API from Windows (run with WSL)

```sh
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

```sh
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

Example of using the API from a WSL (run as a WSL)
```sh
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

Example for using Windows docker desktop.
```sh
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

Avoid ports 8080, 8081, 8082, etc.
> Avoid 8080, 8081, 8082 ports

#### 2nd time and onwards
```sh
docker start hamacho
```

## Set up development environment
## Install dependencies
```sh
pip install -r api/requirements.txt
```
## Set up environment variables
<table>
<tr><td>HOST_DATA_DIR</td><td>Host-side base data directory</td></tr
<tr><td>BASE_DATA_DIR</td><td>Container-side base data directory corresponding to host-side base data directory</td></tr
<tr><td>DBPATH</td><td>path of sqlite3 database file</br> e.g. /tmp/anomary_db.sqlite</td></tr>
<tr><td>PREDICTION_MODE</td><td>SYNC or ASYNC<br
ASYNC is parallel inference in a multi-process environment with torchserve<br>.
SYNC is single-process reasoning with memory bank switching to save memory</td></tr>.
</table>

## Startup
Create an environment with poetry or docker container and execute the following after setting environment variables.
#### development
```sh
python -m api.api
```
#### Production equivalent (example)
```sh
gunicorn --workers 4 --bind 0.0.0.0:80 api.api:app
```

## Tests
Single-process and multi-process tests cannot be done at once, so they are run separately as follows.
To run tests, license files are needed:
- `valid.lic`
- `expired.lic`
Please access [here](https://chowagiken.sharepoint.com/CorporatePlanning/licence-business/Shared%20Documents/Forms/AllItems.aspx?newTargetListUrl=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents&viewpath=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2FForms%2FAllItems%2Easpx&id=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2F02%5F%E7%95%B0%E5%B8%B8%E6%A4%9C%E7%9F%A5%5FAnomalyDetection%2F08%5F%E9%81%8B%E7%94%A8%E4%BF%9D%E5%AE%88%2F01%5F%E9%81%8B%E7%94%A8%2F%E3%83%A9%E3%82%A4%E3%82%BB%E3%83%B3%E3%82%B9%2FPyArmor%2Dlicenses%2Ftests&viewid=d90a7d83%2Dbcee%2D4d76%2D8485%2D0947a8795bdc).

```sh
python -m unittest discover api/tests
python -m unittest discover api/tests/sync
python -m unittest discover api/tests/async
```

## Q&A
- What can I do with the API (compared to the CLI)? 
    - Available via Web API calls from a local PC.
    - Can be used to load weight data into memory in advance so that cold start does not occur for each inference.    

- What the API cannot do (compared to the CLI) 
    - Learning and inference from the command line (except curl command, etc.) 

- How does it work?  
! [data-flow](. /assets/system_image.png)  

- How to use ASYNC and SYNC? 
    - Inference mode SYNC uses less memory and should be used when you want to save memory. 
    - Inference mode ASYNC uses more memory, but it is faster due to parallel processing. 
  
- What should HOST_DATA_DIR refer to? 
    - The top directory of the image files locally.  
     (Note that the path is written differently in Windows and Linux.)

- What should BASE_DATA_DIR_PATH refer to? 
    - The upper directory of image files in the container 
    - The system refers to the path that replaced HOST_DATA_DIR 

- What should I be careful with DBPATH? 
    - The database contains the paths to the image files, so be careful to keep the paths consistent. 

- Operating specs description 
    - Without GPU. 
    - With GPU. (Recommended) 
    - 11th Gen Intel(R) Core(TM) i7 or higher (recommended) 
    - 16GB memory (recommended) 

For details, see [Detailed Design Document](https://chowagiken.sharepoint.com/CorporatePlanning/licence-business/_layouts/15/Doc.aspx?sourcedoc=%7B573E175E-C321-4151-94F0-C15DD200D54F%7D&file=%E7%95%B0%E5%B8%B8%E6%A4%9C%E7%9F%A5AI%E3%82%B7%E3%82%B9%E3%83%86%E3%83%A0%E8%A9%B3%E7%B4%B0%E8%A8AD%E8%A8%88%E6%9B%B8.docx&action=default&mobileredirect=true)

## Sample Code in Python

```python
"""
hamachoが動く環境で
$ pip install -r api/requirements.txt
$ HOST_DATA_DIR=/<parentpath>/anomaly_detection_core_hamacho BASE_DATA_DIR=/app DBPATH=/tmp/anomaly_db.sqlite PREDICTION_MODE=SYNC python -m api.api
でサーバー起動
別のターミナルで以下を実行。
data/bottle-sealing-surface/ 以下にデータがある前提。
$ python demorun.py
"""

import requests
from pathlib import Path
import json

host = "http://localhost:5000/"

good = []
for p in Path("data/bottle-sealing-surface/good").glob("*.png"):
    good.append(str(p))

bad = []
for p in Path("data/bottle-sealing-surface/bad").glob("*.png"):
    bad.append(str(p))

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
model_id = res.json()
print("model_id:", model_id)

res = requests.get(f"{host}/listmodels")
print("models:", res.json())

res = requests.get(f"{host}/modeldetails")
print("models:", res.json())


res = requests.post(f"{host}/servemodel", json={
        "tag": "bottle",
        "mode": "batch",
        "parameters": {
            "project": {
                "save_outputs": {
                    "image": {
                        "classification": ["input_image", "prediction"]
                    },
                    "inference": {
                        "csv": ["anomaly_map", "metrics"]
                    }
                }
            }
        }
    })

res = requests.post(    
    f"{host}/predict", json={"tag": "bottle", "image_paths": good[-2:] + bad[-2:], "save": "all" }
)
print("result:", res.json())

files = [('json', (None, json.dumps({"tag": "bottle", "save": "all"}), "application/json"))]
for im_path in good[-2:] + bad[-2:]:
    filename = Path(im_path).name
    with open(im_path, 'rb') as im:
        files.append(('images', (filename, im.read())))

res =  requests.post("http://localhost:5000/predict?as_file=true", files=files)
print("result with ?as_file=true", res.json())

# list all results
res = requests.get(f"{host}/listresults")
print("all listresults:", res.json())

# list only online mode results of specific model
res = requests.get(f"{host}/listresults?model_id={model_id}&inference_mode=online")
print("online listresults:", res.json())

res = requests.delete(f"{host}/unservemodel?tag=bottle")
print("unserve", res.json())

```