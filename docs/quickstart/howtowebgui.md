# How to use WebGUI

   ![動作イメージ](../assets/hamacho_gui.gif)


## Contents 
- [How to use WebGUI](#how-to-use-webgui)
  - [Contents](#contents)
  - [Simple Starting webgui](#simple-starting-webgui)
      - [Starting from start.bat](#starting-from-startbat)
  - [Starting container](#starting-container)
      - [Use configuration to use API from Windows (run with WSL)](#use-configuration-to-use-api-from-windows-run-with-wsl)
  - [Launching WebGUI](#launching-webgui)


## Simple Starting webgui

#### Starting from start.bat

1. If you are using `Windows`, install `WSL2`.
2. Install `docker ce` on your `WSL` distribution.
3. Download the `docker image` from the following [site](https://chowagiken.sharepoint.com/CorporatePlanning/licence-business/Shared%20Documents/Forms/AllItems.aspx?newTargetListUrl=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents&viewpath=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2FForms%2FAllItems%2Easpx&id=%2FCorporatePlanning%2Flicence%2Dbusiness%2FShared%20Documents%2F02%5F%E7%95%B0%E5%B8%B8%E6%A4%9C%E7%9F%A5%5FAnomalyDetection%2F04%5F%E3%83%97%E3%83%AD%E3%83%80%E3%82%AF%E3%83%88%E9%96%8B%E7%99%BA%5FDevelopment%2F03%5F%E9%96%8B%E7%99%BA%E6%B8%88%E3%81%BF%E6%8F%90%E4%BE%9B%E3%83%A9%E3%82%A4%E3%83%96%E3%83%A9%E3%83%AA%2Fanomaly%5Fdetection%5Fcore%5Fhamacho%2Fv1%2E5%2E0&viewid=d90a7d83%2Dbcee%2D4d76%2D8485%2D0947a8795bdc), and load it with `docker load`. 
4. Download the `visee_inspection` folder located in the same directory.
5. Run `start.bat`.
6. Access http://localhost:3000/.

-------------------------------------------------------------------------

## Starting container


#### Use configuration to use API from Windows (run with WSL)
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

## Launching WebGUI
1. Install `node.js`
2. Rewrite the directory in the file `anomaly_detection_browserUI/context/fields`.ts to your user name
3. Mount `/mnt/c/Users/suzuki/Desktop/anomaly_detection_browserUI-develop/public/results` in `/app/results` and `/app/api/modeldata`
4. execute `npm install`
5. execute `npm run dev`



[[back]](quickstart.md)