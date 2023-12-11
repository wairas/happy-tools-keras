import argparse
import logging
import os
import traceback

import numpy as np

from wai.logging import add_logging_level, set_logging_level
from happy.base.app import init_app
from happy.evaluators import RegressionEvaluator
from happy.models.spectroscopy import create_false_color_image
from happy.preprocessors import Preprocessor, MultiPreprocessor
from happy.region_extractors import FullRegionExtractor
from happy.splitters import HappySplitter
from happy_keras.models.pixel_regression import KerasPixelRegressionModel


PROG = "happy-keras-pixel-regression-build"

logger = logging.getLogger(PROG)


def default_preprocessors() -> str:
    args = [
        "crop -W 320 -H 648",
        "wavelength-subset -f 60 -t 189",
        "sni",
        "snv",
        "derivative -w 15 -d 1",
        "pad -W 320 -H 648 -v 0",
        "down-sample",
    ]
    return " ".join(args)


def main():
    init_app()
    parser = argparse.ArgumentParser(
        description='Evaluate a Keras-based pixel regression model.',
        prog=PROG,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_folder', type=str, help='Path to the data folder', required=True)
    parser.add_argument('-P', '--preprocessors', type=str, help='The preprocessors to apply to the data. Either preprocessor command-line(s) or file with one preprocessor command-line per line.', required=False, default=default_preprocessors())
    parser.add_argument('-t', '--target', type=str, help='Name of the target variable', required=True)
    parser.add_argument('-s', '--happy_splitter_file', type=str, help='Path to JSON file containing splits', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', required=True)
    add_logging_level(parser, short_opt="-V")

    args = parser.parse_args()
    set_logging_level(logger, args.logging_level)

    # Create the output folder if it doesn't exist
    logger.info("Creating output dir: %s" % args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)

    # Create preprocessors
    logger.info("Creating pre-processing")
    preproc = MultiPreprocessor(preprocessor_list=Preprocessor.parse_preprocessors(args.preprocessors))

    # Create a FullRegionExtractor instance
    logger.info("Creating region extractor")
    region_selector = FullRegionExtractor(region_size=None, target_name=args.target)

    # Create a HappySplitter instance
    logger.info("Loading splits: %s" % args.happy_splitter_file)
    happy_splitter = HappySplitter.load_splits_from_json(args.happy_splitter_file)
    train_ids, valid_ids, test_ids = happy_splitter.get_train_validation_test_splits(0, 0)

    # Create a KerasPixelRegressionModel instance
    logger.info("Creating model")
    pixel_regression_model = KerasPixelRegressionModel(
        data_folder=args.data_folder, target=args.target, region_selector=region_selector,
        happy_preprocessor=preproc)

    # Fit the model
    logger.info("Fitting model...")
    pixel_regression_model.fit(id_list=train_ids)

    # Predict using the model
    logger.info("Predicting...")
    predictions, actuals = pixel_regression_model.predict(id_list=test_ids, return_actuals=True)

    evl = RegressionEvaluator(happy_splitter, pixel_regression_model, args.target)
    evl.accumulate_stats(predictions, actuals, 0, 0)
    evl.calculate_and_show_metrics()

    max_actual = np.nanmax(actuals)
    min_actual = np.nanmin(actuals)

    # Save the predictions as PNG images
    for i, prediction in enumerate(predictions):
        if np.isnan(min_actual) or np.isnan(max_actual) or (min_actual == max_actual):
            logger.warning("NaN value detected. Cannot proceed with gradient calculation.")
            continue
        false_color_image = create_false_color_image(prediction, min_actual, max_actual)
        false_color_image.save(os.path.join(args.output_folder, f'false_color_{i}.png'))


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return: 0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
