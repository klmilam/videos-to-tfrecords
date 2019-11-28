from setuptools import find_packages
from setuptools import setup


NAME = "Videos-To-TFRecords"
VERSION = "1.0"
REQUIRED_PACKAGES = ["opencv-python",
                     "google-cloud-storage",
                     "google-resumable-media==0.5.0",
                     "tensorflow_hub>=0.6.0",
                     "tensorflow==2.0.0",
                     "tensorflow-transform"]

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
)
