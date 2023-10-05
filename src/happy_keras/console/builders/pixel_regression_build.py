import argparse
import os
import traceback

import numpy as np

from happy.evaluators import RegressionEvaluator
from happy.model.spectroscopy_model import create_false_color_image
from happy.preprocessors import Preprocessor, MultiPreprocessor
from happy.region_extractors.full_region_extractor import FullRegionExtractor
from happy.splitter.happy_splitter import HappySplitter
from happy_keras.model.pixel_regression_model import KerasPixelRegressionModel


def default_preprocessors() -> str:
    args = [
        "crop -W 320 -H 648"
        "wavelength-subset -f 60 -t 189",
        "sni",
        "snv",
        "derivative -w 15 -d 1",
        "pad -W 320 -H 648 -v 0",
        "down-sample",
    ]
    return " ".join(args)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a Keras-based pixel regression model.',
        prog="happy-keras-pixel-regression-build",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_folder', type=str, help='Path to the data folder', required=True)
    parser.add_argument('-P', '--preprocessors', type=str, help='The preprocessors to apply to the data', required=False, default=default_preprocessors())
    parser.add_argument('-t', '--target', type=str, help='Name of the target variable', required=True)
    parser.add_argument('-s', '--happy_splitter_file', type=str, help='Path to JSON file containing splits', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', required=True)

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Create preprocessors
    preproc = MultiPreprocessor(preprocessor_list=Preprocessor.parse_preprocessors(args.preprocessors))

    # Create a FullRegionSelector instance
    region_selector = FullRegionExtractor(region_size=None, target_name=args.target)

    # Create a HappySplitter instance
    happy_splitter = HappySplitter.load_splits_from_json(args.happy_splitter_file)
    train_ids, valid_ids, test_ids = happy_splitter.get_train_validation_test_splits(0, 0)

    # Create a KerasPixelRegressionModel instance
    pixel_regression_model = KerasPixelRegressionModel(
        data_folder=args.data_folder, target=args.target, region_selector=region_selector,
        happy_preprocessor=preproc)

    # Fit the model
    pixel_regression_model.fit(id_list=train_ids)

    # Predict using the model
    predictions, actuals = pixel_regression_model.predict(id_list=test_ids, return_actuals=True)

    evl = RegressionEvaluator(happy_splitter, pixel_regression_model, args.target)
    evl.accumulate_stats(predictions, actuals, 0, 0)
    evl.calculate_and_show_metrics()

    max_actual = np.nanmax(actuals)
    min_actual = np.nanmin(actuals)

    # Save the predictions as PNG images
    for i, prediction in enumerate(predictions):
        if np.isnan(min_actual) or np.isnan(max_actual) or (min_actual == max_actual):
            print("NaN value detected. Cannot proceed with gradient calculation.")
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
