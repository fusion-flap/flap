from setuptools import setup, find_packages


setup(
    name='flap',
    packages=find_packages(),
    version='1.21.0',
    license='MIT',
    description='Fusion Library of Analysis Programs. This package is intended for analysing large multidimensional datasets.',
    author='Sándor Zoletnik',
    author_email='sandor.zoletnik@ek-cer.hu',
    url='https://github.com/fusion-flap/flap',
    keywords=[
        'fusion',
        'data analysis',
    ],
    install_requires=[
        'h5py >= 3.12.1',
        'ipykernel >= 6.29.5',
        'lxml >= 5.3.0',
        'matplotlib >= 3.9.2',
        'numpy >= 2.1.3',
        'numpydoc >= 1.8.0',
        'opencv-python >= 4.10.0',
        'pandas >= 2.2.3',
        'pickleshare >= 0.7.5',
        'scipy >= 1.14.1',
        'tornado >= 6.4.1',
        'requests >= 2.32.3',
        'pytest >= 8.3.3',
        'sphinx >= 8.1.3',
        'sphinx-copybutton >= 0.5.2',
        'myst-sphinx-gallery >= 0.3.0',
        'myst-parser >= 4.0.0',
        'furo >= 2024.8.6',
        'py2puml >= 0.9.1',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.12.4'
    ],
)
