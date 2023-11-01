
## 前提条件
`docker-compose-gpu.yml` ファイルに記載されているイメージをビルドするにあたって以下の条件が整っていることを確認してください。
- 次の`docker image`をビルドしてください。 `hamacho:<version>-api-prod-obfuscated`  [(ビルド方法について参考)](./manage-docker.jp.md)
- 次の`docker image`をビルドしてください。 `hamacho-webui:<version>`  [(ビルド方法について参考)](https://github.com/chowagiken/anomaly_detection_browserUI/blob/develop/README.md#docker)
- APIが動く `license.lic` を準備してください。

もし上記の`docker image`のバックアップがあるのなら、以下のコマンドでロードしてください。：
```bash
docker load -i <image-file.tar.gz>
```

## WebGUIの起動方法
1. 任意の場所へ次のフォルダをコピーしてください。 (`{project-root}/docker/compose/with_webui`) 
2. `docker-compose-gpu.yml` と `docker-compose-cpu.yml` ファイルを開いて、  
   `core` と `webui` の`docker image`の名称が適切なものかどうかチェックをしてください。異なっていたら、書き換えてください。
3. フォルダ名 `license` を作成し、そこに `license.lic` を入れてください。
4. `start.bat` をダブルクリックして実行してください。
