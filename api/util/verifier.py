import os
import subprocess
import logging
from functools import wraps

logger = logging.getLogger(name=__name__)


class EnvVarError(Exception):
    pass


class LicInconsistencyError(Exception):
    """与えられたライセンスと、難読化に使用されたライセンスの不整合

    例
    - 有償版と無償版が混在している
    """

    pass


def proxy_license_verifier(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            obf_verifier_dummy_dir = os.environ.get("OBFUSCATED_VERIFIER_DUMMY_DIR")
            if obf_verifier_dummy_dir is None:
                logger.error(
                    "Environment variable not found: OBFUSCATED_VERIFIER_DUMMY_DIR"
                )
                raise EnvVarError

            obf_verifier_dummy_path = os.path.join(
                obf_verifier_dummy_dir, "verifier_dummy.py"
            )
            if not os.path.isfile(obf_verifier_dummy_path):
                logger.error("Verifier dummy file not found.")
                raise EnvVarError

            lic_path = os.environ.get("PYARMOR_LICENSE")
            if not lic_path:
                logger.error("Environment variable not found: PYARMOR_LICENSE")
                raise EnvVarError

            if not os.path.isfile(lic_path):
                logger.error("License file not found.")
                raise Exception

            # If license is valid, output will be b'License is valid.\n'.
            # Otherwise, exception is thrown.
            output = subprocess.run(
                ["python", obf_verifier_dummy_path], check=False, capture_output=True
            )

            if output.stderr == b"Check license failed, Invalid input packet.\n":
                raise LicInconsistencyError(
                    "Check license failed, Invalid input packet."
                )

            if output.stdout != b"License is valid.\n":
                logger.error(output)
                raise Exception
        except EnvVarError as e:
            logger.error(e)
            return "application error", 500
        except LicInconsistencyError as e:
            logger.error(e)
            return "application error", 500
        except subprocess.CalledProcessError as e:
            logger.error(e)
            return "License is expired. Please contact publisher.", 401
        except Exception as e:
            logger.error(e)
            return "License is invalid. Please contact publisher.", 401

        r = f(*args, **kwargs)
        return r

    return wrapper
