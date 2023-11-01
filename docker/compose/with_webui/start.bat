@REM Dockerのサービス起動（wsl.confの設定していれば不要かもしれない）
@REM wsl sudo service docker start

@REM ここからHOST_DATA_DIR_PATHを追加するコマンド
@SETLOCAL
@SET CMD=wsl wslpath -w $PWD/data
@FOR /f "DELIMS=" %%A IN ('%CMD%') DO SET HOST_DATA_DIR_PATH_TMP=%%A
SET HOST_DATA_DIR_PATH=%HOST_DATA_DIR_PATH_TMP:\=\\%
echo %HOST_DATA_DIR_PATH%

SET HOST_DATA_DIR_WEBUI_PATH=%HOST_DATA_DIR_PATH_TMP:\=\\\\%
echo %HOST_DATA_DIR_WEBUI_PATH%

SET LIECENSE_DIR=/license/

if not exist ".\results\" mkdir ".\results"
if not exist ".\data\train\" mkdir ".\data\train"
if not exist ".\data\inference\" mkdir ".\data\inference"

echo HOST_DATA_DIR_PATH=%HOST_DATA_DIR_PATH%>.env
echo HOST_DATA_DIR_WEBUI_PATH=%HOST_DATA_DIR_WEBUI_PATH%>>.env
echo APP_DATA_MOUNT_PATH=./data/:/app/data>>.env
echo RESULTS_MOUNT_PATH=./results/:/app/results>>.env
echo WEBUI_RESULTS_MOUNT_PATH=./results/:/app/public/results>>.env
echo MODELDATA_MOUNT_PATH=./results/:/app/api/modeldata/>>.env
echo LIECENSE_MOUNT_PATH=./license/:%LIECENSE_DIR%>>.env
echo PORT=5000>>.env
echo BASE_DATA_DIR_PATH=/app/data>>.env
echo DBPATH=/app/data/visee_db.sqlite>>.env
echo SYNC_MODE=SYNC>>.env

where nvidia-smi.exe >nul 2>nul
if %errorlevel%==0 (
	wsl docker compose -f docker-compose-gpu.yml up
) else (
	wsl docker compose -f docker-compose-cpu.yml up
)
