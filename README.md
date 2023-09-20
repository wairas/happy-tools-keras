# happy-tools-keras
happy-tools that use the Keras library for Deep Learning on hyper-spectral images.

## Installation

```
pip install git+https://github.com/wairas/happy-tools.git
pip install git+https://github.com/wairas/happy-tools-keras.git
```

## Docker

For Docker images, please see [docker/README.md](docker/README.md).

## Commandline

### Keras Pixel Regression Build

```
usage: happy-keras-pixel-regression-build [-h]
                                          data_folder target
                                          happy_splitter_file output_folder

Evaluate a Keras-based pixel regression model.

positional arguments:
  data_folder          Path to the data folder
  target               Name of the target variable
  happy_splitter_file  Path to JSON file containing splits
  output_folder        Path to the output folder

optional arguments:
  -h, --help           show this help message and exit
```


### Keras Segmentation Build

```
usage: keras_segmentation_build.py [-h]
                                   data_folder target happy_splitter_file
                                   output_folder

Build a Keras-based pixel segmentation model.

positional arguments:
  data_folder          Path to the data folder
  target               Name of the target variable
  happy_splitter_file  Path to JSON file containing splits
  output_folder        Path to the output folder

optional arguments:
  -h, --help           show this help message and exit
```


### Keras Unsupervised Build

```
usage: happy-keras-unsupervised-build [-h]
                                      data_folder target happy_splitter_file
                                      output_folder

Build a Keras-based pixel segmentation model.

positional arguments:
  data_folder          Path to the data folder
  target               Name of the target variable
  happy_splitter_file  Path to JSON file containing splits
  output_folder        Path to the output folder

optional arguments:
  -h, --help           show this help message and exit
```
