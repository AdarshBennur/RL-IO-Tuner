"""
Setup script for RL-IO-Tuner
"""

from setuptools import setup, find_packages

setup(
    name="rl-io-tuner",
    version="1.0.0",
    description="Dynamic Linux Storage Optimization Using Deep Reinforcement Learning",
    author="Adarsh Bennur",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "psutil>=5.9.0",
        "tqdm>=4.65.0",
    ],
)
