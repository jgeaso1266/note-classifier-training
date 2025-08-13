from setuptools import find_packages, setup

setup(
    name="my-training",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow",
        "keras",
        "numpy",
        "tqdm",
        "opencv-python"
    ],
)