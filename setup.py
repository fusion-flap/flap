from setuptools import setup, find_packages


setup(
    name = 'flap',
    packages = find_packages(),
    version = '1.12.0',
    license = 'MIT',
    description = 'Fusion Library of Analysis Programs This package is intended for analysing large multidimensional datasets.',
    author = 'Sándor Zoletnik',
    author_email = 'sandor.zoletnik@ek-cer.hu',
    url = 'https://github.com/fusion-flap/flap',
    keywords = [
        'fusion',
        'data analysis',
    ],
    install_requires = [
        'h5py >= 3.6.0',
        'ipykernel >= 6.9.2',
        'lxml >= 4.8.0',
        'matplotlib >= 3.5.1',
        'numpy >= 1.22.3',
        'opencv-python >= 4.5.5',
        'pandas >= 1.4.1',
        'pickleshare >= 0.7.5',
        'scipy >= 1.8.0',
        'tornado >= 6.1'
    ],
    classifiers = [
        'Programming Language :: Python :: 3.9.10'
    ],
)