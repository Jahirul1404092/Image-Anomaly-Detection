import os
import sysconfig
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize

EXCLUDE_FILES = ["./setup.py"]

class touchinit:
    def __init__(self, rootpath):
        self.rootpath = Path(rootpath)
        self.l_path = []
    def __enter__(self):
        for p in self.rootpath.rglob('*'):
            if p.is_dir() and not (p / '__init__.py').exists():
                (p / '__init__.py').touch()
                self.l_path.append((p / '__init__.py'))
    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.l_path:
            p.unlink()

class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = []
        for (pkg, mod, filepath) in modules:
            if os.path.exists(filepath.replace(".py", ext_suffix)):
                continue
            filtered_modules.append((pkg, mod, filepath,))
        return filtered_modules


def get_ext_paths(root_dir, exclude_files):
    """get filepaths for compilation"""
    paths = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != ".py":
                continue
            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue
            paths.append(file_path)
    return paths

with touchinit('./'):
    setup(
        name="app",
        version="1.0.0",
        packages=find_packages(),
        ext_modules=cythonize(get_ext_paths(".", EXCLUDE_FILES), compiler_directives={
            "language_level": 3, 
            "always_allow_keywords": True,
            "annotation_typing": False}),
        cmdclass={"build_py": build_py},
    )