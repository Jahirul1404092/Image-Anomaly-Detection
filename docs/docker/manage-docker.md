
# Prerequisites

- Latest NVIDIA Driver (if the machine has a NVIDIA CUDA capable GPU)
- Latest Docker

## Installation
- Uninstall Docker Desktop.
- Install docker in your WSL distro. [(follow this guide)](https://dev.to/bowmanjd/install-docker-on-windows-wsl-without-docker-desktop-34m9)
    - You can also follow the [official guide](https://docs.docker.com/engine/install/ubuntu/).
- Install nvidia-docker2 if you have a NVIDIA GPU. [(official guide)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).

## Docker Builds
Run `./docker/build_image.sh` from project root.
You should see something like below:
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

The script needs to be prepended by `sudo` if Docker was installed for root only. Example, `sudo ./docker/build_image.sh`.

#### Export docker image:
```bash
docker save <image>:<tag> -o <export-name>.tar.gz
# example
docker save hamacho:v1.4.0-prod-api-obfuscated -o hmc-1.4-prod-obfuscated.tar.gz
```

#### Import docker image from exported file:
```bash
docker load -i <path-to-docker-image-file>
# example
docker load -i hmc-1.4-prod-obfuscated.tar.gz
```

***************************

## Running Containers

### If you want to sync (mount) a local folder and run train/inference

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

### For development

```bash
docker run  -it \
            --rm \           # Optional, it will auto delete the container when you stop it
            --name hmc-dev \ # You can give it a different name
            --gpus all \     # If the machine has GPU
            -v C:\repo\anomaly_detection_core_hamacho:/hmc \
                             # Mount the host dataset folder to `/hmc`
            -w /hmc \        # Must be the same as mounted folder in container
            hamacho:dev \    # In my case the image name and tag is hamacho:dev. In your case it might be different
            bash
```
