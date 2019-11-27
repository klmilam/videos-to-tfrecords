from setuptools import find_packages
from setuptools import setup


NAME = "Videos-To-TFRecords"
VERSION = "1.0"
REQUIRED_PACKAGES = ["opencv-python",
                     "google-cloud-storage",
                     "google-resumable-media==0.5.0"]

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
)
