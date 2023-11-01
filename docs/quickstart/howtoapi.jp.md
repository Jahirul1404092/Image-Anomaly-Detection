# APIの使用方法

## 目次 
- [docker imageの作成](#cythonによる共有ライブラリ化とコンテナイメージ作成)
- [containerの起動](#コンテナ起動)
- [windowsからAPIを使用](#windowsからpythonでapiを利用する方法-wslで実行)


次の構造でデータセットを準備します。

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

## 本番環境のデプロイ

### Cythonによる共有ライブラリ化とコンテナイメージ作成

(リポジトリルートで以下のコマンドを実行)
```
./docker/build_image.sh     # <-- select prod-api-obfuscated
```

### 開発・テスト環境向け
```
./docker/build_image.sh     # <-- select api or api-test
```

## コンテナ起動
OSによってAPIを利用する****HOST_DATA_DIR****のパス設定が異なる  
※Licenseファイルもマウントする
`hamacho:v1.4.0-prod-api-obfuscated`に対して、
 `hamacho:v1.4.0-api-test` と `hamacho:v1.4.0-api` では  
  要求する引数も異なりますので注意してください。
#### WindowsからAPIを利用する場合の例 (WSLで実行)
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

#### WSLからAPIを利用する場合の例 (WSLで実行)
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
#### Windows docker desktopを利用する場合の例
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
#### コンテナがある状態で2度目以降の起動方法
```
docker start hamacho
```

## WindowsからPythonでAPIを利用する方法 (WSLで実行)
cmdから次のようなPythonでAPIを実行する



```python
"""
サーバー起動
別のターミナルで以下を実行。
data/bottle/ 以下にデータがある前提。
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

### モデル削除
```sh
curl -X DELETE http://localhost:5000/delmodel -H "Content-Type: application/json" -d "[3]"
```

### モデル登録解除
```sh
 curl -X DELETE http://localhost:5000/unservemodel?model_id=1
```

### 登録画像削除
```sh
curl -X DELETE http://localhost:5000/delimage -H "Content-Type: application/json" -d '{"image_tag":"bottle"}'
```





詳細は[こちら](./api/manage-api.jp.md)を参照


[Quick Start Guideに戻る](quickstart.jp.md)