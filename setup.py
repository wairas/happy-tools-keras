from setuptools import setup, find_namespace_packages


def _read(f):
    """
    Reads in the content of the file.
    :param f: the file to read
    :type f: str
    :return: the content
    :rtype: str
    """
    return open(f, 'rb').read()


setup(
    name="happy_tools_keras",
    description="happy-tools that use the Keras library for Deep Learning on hyperspectral images.",
    long_description=(
        _read('DESCRIPTION.rst') + b'\n' +
        _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/wairas/happy-tools-keras",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
    ],
    license='MIT License',
    install_requires=[
        "happy_tools",
        "tensorflow==2.11.0",
        "matplotlib",
    ],
    package_dir={
        '': 'src'
    },
    packages=find_namespace_packages(where='src'),
    entry_points={
        "console_scripts": [
            "happy-generic-keras-pixel-regression-build=happy_keras.console.builders.generic_pixel_regression_build:sys_main",
            "happy-generic-keras-segmentation-build=happy_keras.console.builders.generic_segmentation_build:sys_main",
            "happy-generic-keras-unsupervised-build=happy_keras.console.builders.generic_unsupervised_build:sys_main",
            "happy-keras-pixel-regression-build=happy_keras.console.builders.pixel_regression_build:sys_main",
            "happy-keras-segmentation-build=happy_keras.console.builders.segmentation_build:sys_main",
            "happy-keras-unsupervised-build=happy_keras.console.builders.unsupervised_build:sys_main",
        ],
        "class_lister": [
            "happy=happy_keras.class_lister",
        ],
    },
    version="0.0.1",
    author='Dale Fletcher',
    author_email='dale@waikato.ac.nz',
)
