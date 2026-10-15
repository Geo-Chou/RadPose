from setuptools import find_packages, setup

setup(
    name="radpose",
    version="1.0.0",
    packages=find_packages(include=("mmwave3d", "mmwave3d.*")),
    install_requires=[
        "mmcv-lite>=2.0.0rc4,<2.2.0",
        "mmengine>=0.8.0,<1.0.0",
        "numpy",
        "scipy",
        "spconv-cu118",
        "torch>=2.0",
    ],
)
