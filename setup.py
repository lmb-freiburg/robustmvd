from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open(path.join(here, 'rmvd', 'version.py')).read())
setup(
    name='rmvd',
    version=__version__,
    description='Robust Multi-view Depth Framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/robustmvd/robustmvd',
    author='Philipp Schröppel',
    author_email='schroepp@cs.uni-freiburg.de',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Programming Language :: Python :: 3.15',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='pytorch models datatasets depth estimation multi-view benchmark',
    packages=find_packages(exclude=['sample_data', 'tests', 'weights']),
    include_package_data=True,
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pillow',
        'matplotlib',
        'pandas',
        'pytoml',
        'tqdm',
        'opencv-python-headless',
        'scikit-image',
        'pyqt5',
        'dill',
        'easydict',
        'wandb',
        'tensorboard',
        'gdown',
        'termcolor',
        'einops',
        'itypes @ git+https://github.com/pschroeppel/itypes.git',
        'iutils @ git+https://github.com/pschroeppel/iutils.git',
        'iviz @ git+https://github.com/pschroeppel/iviz.git',
        'imetrics @ git+https://github.com/pschroeppel/imetrics.git',
    ],
    python_requires='>=3.10',
)
