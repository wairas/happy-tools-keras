# happy-tools-keras
happy-tools that use the Keras library for Deep Learning on hyperspectral images.

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
usage: happy-keras-pixel-regression-build [-h] -d DATA_FOLDER -t TARGET -s
                                          HAPPY_SPLITTER_FILE -o OUTPUT_FOLDER

Evaluate a Keras-based pixel regression model.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Path to the data folder (default: None)
  -t TARGET, --target TARGET
                        Name of the target variable (default: None)
  -s HAPPY_SPLITTER_FILE, --happy_splitter_file HAPPY_SPLITTER_FILE
                        Path to JSON file containing splits (default: None)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Path to the output folder (default: None)
```


### Keras Segmentation Build

```
usage: happy-keras-segmentation-build [-h] -d DATA_FOLDER -t TARGET
                                      [-n NUM_CLASSES] -s HAPPY_SPLITTER_FILE
                                      -o OUTPUT_FOLDER

Build a Keras-based pixel segmentation model.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Path to the data folder (default: None)
  -t TARGET, --target TARGET
                        Name of the target variable (default: None)
  -n NUM_CLASSES, --num_classes NUM_CLASSES
                        The number of classes, used for generating the mapping
                        (default: 4)
  -s HAPPY_SPLITTER_FILE, --happy_splitter_file HAPPY_SPLITTER_FILE
                        Path to JSON file containing splits (default: None)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Path to the output folder (default: None)
```


### Keras Unsupervised Build

```
usage: happy-keras-unsupervised-build [-h] -d DATA_FOLDER -t TARGET
                                      [-n NUM_CLUSTERS] -s HAPPY_SPLITTER_FILE
                                      -o OUTPUT_FOLDER

Build a Keras-based unsupervised segmentation model.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Path to the data folder (default: None)
  -t TARGET, --target TARGET
                        Name of the target variable (default: None)
  -n NUM_CLUSTERS, --num_clusters NUM_CLUSTERS
                        The number of clusters to use (default: 4)
  -s HAPPY_SPLITTER_FILE, --happy_splitter_file HAPPY_SPLITTER_FILE
                        Path to JSON file containing splits (default: None)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Path to the output folder (default: None)
```
