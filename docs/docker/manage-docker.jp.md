# 前提条件

- 最新のNVIDIAドライバ（NVIDIA CUDA対応GPUを搭載したマシンの場合）
- 最新のDocker

# インストール
- `Docker Desktop`をアンインストールしてください。
- `wsl` に `docker ce` を 次のガイドにしたがってインストールしてください。[(ガイド)](https://dev.to/bowmanjd/install-docker-on-windows-wsl-without-docker-desktop-34m9) 
    - 公式ガイドも参照してください[(公式ガイド)](https://docs.docker.com/engine/install/ubuntu/)
- 動作環境に`NVIDA GPU` があれば `nvidia-docker2`をインストールして下さい。[(公式ガイド)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)


## Docker ビルド
プロジェクトルートから `./docker/build_image.sh` を実行します。  
以下のように表示されるはずです:
```bash
$ ./docker/build_image.sh
Build the desired docker image from the options below:
0) prod
1) dev
2) api
3) test
4) api-test
5) prod-lib
6) prod-api-obfuscated
===========================
Example: ./build_image.sh
Selection: 1
===========================
Detected version from 'pyproject.toml' file is 1.4.0
Select image type:
Selection: # <-- your input here, here i have selected 0

The <image>:<tag> will be hamacho:v1.4.0-prod
If a different <image>:<tag> is required, please input it below. Otherwise hit Enter.
===========================
Example:
<image>:<tag> => abd:v1.0
===========================
<image>:<tag> => # <-- your custom image name and tag here. If you want to keep it as is, you can just hit Enter.
```

Dockerをroot権限でインストールした場合は、スクリプトの前に`sudo`を付ける必要があります。例：`sudo ./docker/build_image.sh`.

#### docker image の保存方法
```bash
docker save <image>:<tag> -o <export-name>.tar.gz
# example
docker save hamacho:v1.4.0-prod-api-obfuscated -o hmc-1.4-prod-obfuscated.tar.gz
```

#### docker image の読込方法
```bash
docker load -i <path-to-docker-image-file>
# example
docker load -i hmc-1.4-prod-obfuscated.tar.gz
```


***************************




## コンテナの実行方法

### ローカルフォルダを同期（マウント）してtrain/inferenceを実行したい場合

```bash
docker run  -it \
            --rm \           # Optional, it will auto delete the container when you stop it
            --name hmc \     # You can give it a different name
            --gpus all \     # If the machine has GPU
            -v C:\repo\anomaly_detection_core_hamacho\data:/app/data \
                             # Mount the host dataset folder to `/app/data`
            -v C:\repo\anomaly_detection_core_hamacho\results:/app/results \
                             # Mount the host results folder to `/app/results`
            hamacho:prod \   # In my case the image name and tag is hamacho:prod. In your case it might be different
            bash
```

### 開発用

```bash
docker run  -it \
            --rm \           # オプションで、コンテナを停止すると自動で削除されます。
            --name hmc-dev \ # 別の名前をつけることも可能です
            --gpus all \     # マシンにGPUがある場合
            -v C:\repo\anomaly_detection_core_hamacho:/hmc \
                             # ホストデータセットフォルダを `/hmc` にマウントする。
            -w /hmc \        # コンテナ内にマウントされたフォルダと同じである必要があります。
            hamacho:dev \    # 私の場合、画像名とタグはhamacho:devになっています。あなたの場合は違うかもしれません
            bash
```
