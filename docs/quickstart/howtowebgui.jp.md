# WebGUIの使用方法

   ![動作イメージ](../assets/hamacho_gui.gif)


## 目次 
- [WebGUIの使用方法](#webguiの使用方法)
  - [目次](#目次)
  - [簡単起動](#簡単起動)
      - [start.batからの起動](#startbatからの起動)
  - [コンテナの起動](#コンテナの起動)
      - [WindowsからAPIを利用する設定を利用 (WSLで実行)](#windowsからapiを利用する設定を利用-wslで実行)
  - [WebGUIの起動](#webguiの起動)


## 簡単起動

#### start.batからの起動

1. `Windows`の場合は`WSL2`をインストールする
2. WSLのディストリビューションに`docker ce`をインストールする
3. 次の[サイト](https://chowagiken.sharepoint.com/CorporatePlanning/licence-business/Shared%20Documents/Forms/AllItems.aspx?newTargetListUrl=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents&viewpath=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2FForms%2FAllItems%2Easpx&id=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2F02%5F%E7%95%B0%E5%B8%B8%E6%A4%9C%E7%9F%A5%5FAnomalyDetection%2F04%5F%E3%83%97%E3%83%AD%E3%83%80%E3%82%AF%E3%83%88%E9%96%8B%E7%99%BA%5FDevelopment%2F03%5F%E9%96%8B%E7%99%BA%E6%B8%88%E3%81%BF%E6%8F%90%E4%BE%9B%E3%83%A9%E3%82%A4%E3%83%96%E3%83%A9%E3%83%AA%2Fanomaly%5Fdetection%5Fcore%5Fhamacho%2Fv1%2E5%2E0&viewid=d90a7d83%2Dbcee%2D4d76%2D8485%2D0947a8795bdc)から`docker image`をダウンロードして、`docker load`で読み込む。
4. 同フォルダ内にある`visee_inspection`フォルダをダウンロードする
5. `start.bat`を実行する
6. http://localhost:3000/ にアクセスする

-------------------------------------------------------------------------
## コンテナの起動

#### WindowsからAPIを利用する設定を利用 (WSLで実行)
```
docker run --name visee-anodet-core \
-e HOST_DATA_DIR=C:\\\\Users\\\\suzuki\\\\Desktop\\\\visee_Inspection\\\\data \
-e BASE_DATA_DIR=/data \
-v /mnt/c/Users/suzuki/Desktop//visee_Inspection/data/:/data \
-v /mnt/c/Users/suzuki/Desktop//visee_Inspection/license/:/license/ \
-e DBPATH=/app/visee_db.sqlite \
-e PREDICTION_MODE=SYNC \
-p 5000:5000 \
-v /mnt/c/Users/suzuki/Desktop//visee_Inspection/results/:/app/results \
-v /mnt/c/Users/suzuki/Desktop//visee_Inspection/results/:/app/api/modeldata/ \
-e PORT=5000 \
visee-anodet-core:v1.4.0

```

## WebGUIの起動

1. `node.js`のインストール
2. `anomaly_detection_browserUI/context/fields.ts`のファイル中のディレクトリを自身のユーザー名に書き換える
3. `/app/results`と`/app/api/modeldata`に`/app/anomaly_detection_browserUI-develop\public\results`をマウントする
4. `npm install`
5. `npm run dev`


[Quick Start Guideに戻る](quickstart.jp.md)