from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        "hiper",
        "hiper.vio",
        "hiper.segmentation",
        "hiper.ephemerality",
        "hiper.object_encoder",
        "hiper.map_io",
        "hiper.common",
        "hiper.evaluation",
    ],
    package_dir={"hiper": "src"},
)

setup(**d)
