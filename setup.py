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


#COMPILE_FLAGS = ['-Ofast','-O3', '-march=native', '-std=c99', '-fopenmp']
COMPILE_FLAGS = ['-O3', '-march=native', '-std=c99', '-fopenmp']
LINK_FLAGS = ['-fopenmp']
MACROS = [
    ('BURST_VERSION_MAJOR', VERSION_MAJOR),
    ('BURST_VERSION_MINOR', VERSION_MINOR),
    ('BURST_VERSION_POINT', VERSION_POINT),
    ('BURST_DM_NOTRANSPOSE', '1'),
    ]

LIBRARY_DIRS = []
INCLUDE_DIRS = []


ext_dedisperse = Extension(
    "burst_search.dedisperse",
    ["burst_search/dedisperse.pyx", "src/dedisperse.c",
        "src/dedisperse_gbt.c"],
    include_dirs=INCLUDE_DIRS + [np.get_include(), "src/"],
    library_dirs = LIBRARY_DIRS,
    depends=["src/dedisperse.h", "dedisperse_gbt.h"],
    extra_compile_args=COMPILE_FLAGS,
    extra_link_args=LINK_FLAGS,
    define_macros=MACROS,
    )

ext_search = Extension(
    "burst_search._search",
    ["burst_search/_search.pyx", "src/dedisperse_gbt.c"],
    include_dirs=INCLUDE_DIRS + [np.get_include(), "src/"],
    library_dirs = LIBRARY_DIRS,
    depends=["dedisperse_gbt.h"],
    extra_compile_args=COMPILE_FLAGS,
    extra_link_args=LINK_FLAGS,
    define_macros=MACROS,
    )

ext_preprocess = Extension(
    "burst_search._preprocess",
    ["burst_search/_preprocess.pyx", "src/dedisperse_gbt.c", "src/preprocess.c"],
    include_dirs=INCLUDE_DIRS + [np.get_include(), "src/"],
    library_dirs = LIBRARY_DIRS,
    depends=["dedisperse_gbt.h", "preprocess.h"],
    extra_compile_args=COMPILE_FLAGS,
    extra_link_args=LINK_FLAGS,
    define_macros=MACROS,
    )


EXTENSIONS = [ext_dedisperse, ext_search, ext_preprocess]


SCRIPTS = ["scripts/burst_guppi", "scripts/burst_watch_guppi", "scripts/burst_bench"]


setup(
    name = 'burst_search',
    version = VERSION,
    packages = ['burst_search'],
    ext_modules = EXTENSIONS,
    scripts = SCRIPTS,
    cmdclass = {'build_ext': build_ext},
    install_requires = ['numpy', 'astropy', 'Cython', 'h5py', 'matplotlib'],

    # metadata for upload to PyPI
    author = "Kiyoshi Wesley Masui, Jonathan Sievers",
    author_email = "kiyo@physics.ubc.ca",
    description = "Fast radio burst search software.",
    license = "GPL v2.0",
    url = "http://github.com/kiyo-masui/burst_search"
)

