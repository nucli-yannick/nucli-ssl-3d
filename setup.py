from __future__ import annotations

from setuptools import setup, find_packages


requires = ['torch>=1.9', 'torchvision', 'nibabel', 'numpy', 'scipy', 'scikit-image','tqdm', 'matplotlib', 'seaborn', 'torchmetrics', 'onnx', 'mlflow', 'indexed-gzip', 'blosc2', 'monai']
setup(
    name="ssl-3d",  # Just a placeholder name
    version="0.1.0",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires=">=3.7.0",
    install_requires=requires
)