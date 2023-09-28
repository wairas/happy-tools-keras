import numpy as np
import argparse
import os
import traceback

from happy.preprocessors.preprocessors import SpectralNoiseInterpolator, PadPreprocessor, SNVPreprocessor, \
    MultiPreprocessor, DerivativePreprocessor, WavelengthSubsetPreprocessor, CropPreprocessor, DownsamplePreprocessor
from happy.splitter.happy_splitter import HappySplitter
from happy.model.spectroscopy_model import create_false_color_image
from happy_keras.model.pixel_regression_model import KerasPixelRegressionModel
from happy.evaluators.regression_evaluator import RegressionEvaluator
from happy.region_extractors.full_region_extractor import FullRegionExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a Keras-based pixel regression model.',
        prog="happy-keras-pixel-regression-build",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_folder', type=str, help='Path to the data folder', required=True)
    parser.add_argument('-t', '--target', type=str, help='Name of the target variable', required=True)
    parser.add_argument('-s', '--happy_splitter_file', type=str, help='Path to JSON file containing splits', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', required=True)

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Create preprocessors
    ppp = PadPreprocessor(width=320, height=648, pad_value=0)
    subset_indices = list(range(60, 190))
    w = WavelengthSubsetPreprocessor(subset_indices=subset_indices)
    crop = CropPreprocessor(width=320, height=648, pad=False)
    clean = SpectralNoiseInterpolator()
    snv = SNVPreprocessor()
    sg = DerivativePreprocessor(window_length=15, deriv=1)
    ds = DownsamplePreprocessor()
    multi = MultiPreprocessor(preprocessor_list=[crop, w, clean, snv, sg, ppp, ds])

    # Create a FullRegionSelector instance
    region_selector = FullRegionExtractor(region_size=None, target_name=args.target)

    # Create a HappySplitter instance
    happy_splitter = HappySplitter.load_splits_from_json(args.happy_splitter_file)
    train_ids, valid_ids, test_ids = happy_splitter.get_train_validation_test_splits(0, 0)

    # Create a KerasPixelRegressionModel instance
    pixel_regression_model = KerasPixelRegressionModel(
        data_folder=args.data_folder, target=args.target, region_selector=region_selector,
        happy_preprocessor=multi)

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
