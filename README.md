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
usage: happy-keras-pixel-regression-build [-h] -d DATA_FOLDER
                                          [-P PREPROCESSORS] -t TARGET -s
                                          HAPPY_SPLITTER_FILE -o OUTPUT_FOLDER
                                          [-V {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Evaluate a Keras-based pixel regression model.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Path to the data folder (default: None)
  -P PREPROCESSORS, --preprocessors PREPROCESSORS
                        The preprocessors to apply to the data. Either
                        preprocessor command-line(s) or file with one
                        preprocessor command-line per line. (default: crop -W
                        320 -H 648 wavelength-subset -f 60 -t 189 sni snv
                        derivative -w 15 -d 1 pad -W 320 -H 648 -v 0 down-
                        sample)
  -t TARGET, --target TARGET
                        Name of the target variable (default: None)
  -s HAPPY_SPLITTER_FILE, --happy_splitter_file HAPPY_SPLITTER_FILE
                        Path to JSON file containing splits (default: None)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Path to the output folder (default: None)
  -V {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
```


### Keras Segmentation Build

```
usage: happy-keras-segmentation-build [-h] -d DATA_FOLDER [-P PREPROCESSORS]
                                      -t TARGET -s HAPPY_SPLITTER_FILE -o
                                      OUTPUT_FOLDER
                                      [-V {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Build a Keras-based pixel segmentation model.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Path to the data folder (default: None)
  -P PREPROCESSORS, --preprocessors PREPROCESSORS
                        The preprocessors to apply to the data. Either
                        preprocessor command-line(s) or file with one
                        preprocessor command-line per line. (default:
                        wavelength-subset -f 60 -t 189 sni snv derivative -w
                        15 -d 1 pad -W 128 -H 128 -v 0)
  -t TARGET, --target TARGET
                        Name of the target variable (default: None)
  -s HAPPY_SPLITTER_FILE, --happy_splitter_file HAPPY_SPLITTER_FILE
                        Path to JSON file containing splits (default: None)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Path to the output folder (default: None)
  -V {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
```


### Keras Unsupervised Build

```
usage: happy-keras-unsupervised-build [-h] -d DATA_FOLDER [-P PREPROCESSORS]
                                      -t TARGET [-n NUM_CLUSTERS] -s
                                      HAPPY_SPLITTER_FILE -o OUTPUT_FOLDER
                                      [-V {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Build a Keras-based unsupervised segmentation model.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Path to the data folder (default: None)
  -P PREPROCESSORS, --preprocessors PREPROCESSORS
                        The preprocessors to apply to the data. Either
                        preprocessor command-line(s) or file with one
                        preprocessor command-line per line. (default:
                        wavelength-subset -f 60 -t 189 snv derivative pad -W
                        128 -H 128 -v 0)
  -t TARGET, --target TARGET
                        Name of the target variable (default: None)
  -n NUM_CLUSTERS, --num_clusters NUM_CLUSTERS
                        The number of clusters to use (default: 4)
  -s HAPPY_SPLITTER_FILE, --happy_splitter_file HAPPY_SPLITTER_FILE
                        Path to JSON file containing splits (default: None)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Path to the output folder (default: None)
  -V {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
```
