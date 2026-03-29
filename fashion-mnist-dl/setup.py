# setup.py — makes src/ pip-installable as a local package
# Run: pip install -e .
# After this, notebooks can do: from src.models import build_regularized_cnn

from setuptools import setup, find_packages

setup(
    name="fashion-mnist-dl",
    version="0.1.0",
    description="End-to-End Deep Learning Pipeline on Fashion-MNIST",
    author="Anshum",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "Pillow>=9.4.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "ydata-profiling>=4.5.0",
            "ipywidgets>=8.0.0",
        ]
    },
)
