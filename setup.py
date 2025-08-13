from setuptools import find_packages, setup

setup(
    name="my-training",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.10.0",
        "keras<2.14.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
        "opencv-python"
    ],
)