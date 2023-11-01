
## Prerequisites
Build the images mentioned in the `docker-compose-gpu.yml` file.
- Build `hamacho:<version>-api-prod-obfuscated` docker image. [(follow this guide)](./manage-docker.md)
- Build `hamacho-webui:<version>` docker image. [(follow this guide)](https://github.com/chowagiken/anomaly_detection_browserUI/blob/develop/README.en.md#docker)
- Collect the `license.lic` file.

If you have image backups of the image(s) mentioned above, import the images:
```bash
docker load -i <image-file.tar.gz>
```

## How to execute
1. Copy the contents of this directory (`{project-root}/docker/compose/with_webui`) to the desired location.
2. Open the `docker-compose-gpu.yml` & `docker-compose-cpu.yml` file.
    Check and make sure if the image name (`image:`) of the `core` and `webui` service is correct or not.
3. Create a folder named `license` and copy the `license.lic` file you collected into this folder.
4. Double-click `start.bat` file.
