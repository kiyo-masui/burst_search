from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_POINT = 0
VERSION_DEV = 1

VERSION = "%d.%d.%d" % (VERSION_MAJOR, VERSION_MINOR, VERSION_POINT)
if VERSION_DEV:
    VERSION = VERSION + ".dev%d" % VERSION_DEV


COMPILE_FLAGS = ['-Ofast', '-march=native', '-std=c99', '-fopenmp']
MACROS = [
    ('BURST_VERSION_MAJOR', VERSION_MAJOR),
    ('BURST_VERSION_MINOR', VERSION_MINOR),
    ('BURST_VERSION_POINT', VERSION_POINT),
    ]

LIBRARY_DIRS = []
INCLUDE_DIRS = []


ext_dedisperse = Extension(
    "burst_search.dedisperse",
    ["burst_search/dedisperse.pyx", "src/dedisperse.c",],
    include_dirs=INCLUDE_DIRS + [np.get_include(), "src/"],
    library_dirs = LIBRARY_DIRS,
    #depends=["src/dedisperse.h",],
    extra_compile_args=COMPILE_FLAGS,
    define_macros=MACROS,
    )


EXTENSIONS = [ext_dedisperse,]


setup(
    name = 'burst_search',
    version = VERSION,

    packages = ['burst_search'],
    scripts=[],
    ext_modules = EXTENSIONS,
    cmdclass = {'build_ext': build_ext},
    install_requires = ['numpy', 'pyfits', 'Cython'],

    # metadata for upload to PyPI
    author = "Kiyoshi Wesley Masui, Jonathan Sievers",
    author_email = "kiyo@physics.ubc.ca",
    description = "Fast radio burst search software.",
    license = "GPL v2.0",
    url = "http://github.com/kiyo-masui/burst_search"
)

