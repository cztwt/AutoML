from setuptools import setup, Extension
from Cython.Build import cythonize

"""
# LDFLAGS="-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib" \
#   python build_pylib.py build_ext
setup(ext_modules=cythonize([
        'feutils.py',
        'fewindows.py',
        'ioutils.py',
        'pyfe_standalone.py'
    ],
    build_dir='./build/cfiles'
))
"""

extensions = [
    Extension(name='easyai.fe.fefuncs',
        sources=['/Users/chenzhao/Desktop/AutoML/cython_test/sumwhere.py'],
        library_dirs=['/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib']
    ),
]


# 自动生成的c文件放在cfiles文件夹下
setup(
    name="easyai",
    version="0.1.0",
    ext_modules=cythonize(extensions, build_dir='./build/cfiles')
)