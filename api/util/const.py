import os
import re

class Const:
    model_version = 1
    version_prefix = ''
    save_dir = 'api/modeldata'
    config_path = 'api/config.yaml'

def win2posix(host):
    base_dir = os.environ.get('BASE_DATA_DIR', '')
    host_dir = os.environ.get('HOST_DATA_DIR', '')
    base = re.sub(f"^{host_dir}", base_dir, host)
    if re.match(r'^[A-Za-z]:', host_dir):
        base = re.sub(r'\\', '/', base)
    return base
