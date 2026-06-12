from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        "seliSlam",
        "seliSlam.vio",
        "seliSlam.segmentation",
        "seliSlam.ephemerality",
        "seliSlam.object_encoder",
        "seliSlam.map_io",
        "seliSlam.common",
    ],
    package_dir={"seliSlam": "src"},
)

setup(**d)
