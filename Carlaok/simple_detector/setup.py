import sys
import pcl

try:
    from skbuild import setup
except ImportError:
    raise ImportError('scikit-build is required for installing')

setup(
    name="Carla Detector",
    version="0.0.1",
    description="Detector for carla",
    author='Jacob Zhong',
    author_email='cmpute@gmail.com',
    license='BSD-3-Clause',
)
