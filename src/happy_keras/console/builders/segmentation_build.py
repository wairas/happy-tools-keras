import argparse
import os
import traceback

from happy.evaluators.classification_evaluator import ClassificationEvaluator
from happy.preprocessors.preprocessors import SpectralNoiseInterpolator, PadPreprocessor, SNVPreprocessor, \
    MultiPreprocessor, DerivativePreprocessor, WavelengthSubsetPreprocessor
from happy.region_extractors.full_region_extractor import FullRegionExtractor
from happy.splitter.happy_splitter import HappySplitter
from happy_keras.model.segmentation_model import KerasPixelSegmentationModel, create_false_color_image, create_prediction_image


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Build a Keras-based pixel segmentation model.',
        prog="happy-keras-segmentation-build",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_folder', type=str, help='Path to the data folder', required=True)
    parser.add_argument('-t', '--target', type=str, help='Name of the target variable', required=True)
    parser.add_argument('-s', '--happy_splitter_file', type=str, help='Path to JSON file containing splits', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', required=True)

    args = parser.parse_args()

    # there is an optional mapping file in happy data now, but TODO here.
    mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 3
        # Add more classes and their corresponding integer labels here
    }

    # preprocessing
    pp = PadPreprocessor(width=128, height=128, pad_value=0)
    subset_indices = list(range(60, 190))
    w = WavelengthSubsetPreprocessor(subset_indices=subset_indices)
    clean = SpectralNoiseInterpolator()
    SNVpp = SNVPreprocessor()
    SGpp = DerivativePreprocessor(window_length=15, deriv=1)
    pp = MultiPreprocessor(preprocessor_list=[w, clean, SNVpp, SGpp, pp])
    
    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Create a FullRegionSelector instance
    region_selector = FullRegionExtractor(region_size=None, target_name=args.target)
    happy_splitter = HappySplitter.load_splits_from_json(args.happy_splitter_file)
    train_ids, valid_ids, test_ids = happy_splitter.get_train_validation_test_splits(0,0)

    # Create a KerasPixelSegmentationModel instance
    pixel_segmentation_model = KerasPixelSegmentationModel(data_folder=args.data_folder, target=args.target, region_selector=region_selector, mapping=mapping, happy_preprocessor=pp)

    # Load sample IDs (you can modify this based on your folder structure)
    #sample_ids = [f.name for f in os.scandir(args.data_folder) if f.is_dir()]

    # Fit the model
    pixel_segmentation_model.fit(id_list=train_ids, target_variable=args.target)
    
    # Predict using the model
    predictions, actuals = pixel_segmentation_model.predict(id_list=test_ids, return_actuals=True)
    evl = ClassificationEvaluator(happy_splitter, pixel_segmentation_model, args.target)
    evl.accumulate_stats(predictions, actuals, 0, 0)
    evl.calculate_and_show_metrics()

    # Save the predictions as PNG images
    for i, prediction in enumerate(predictions):
        prediction_image = create_prediction_image(prediction)
        prediction_image.save(os.path.join(args.output_folder, f'prediction_{i}.png'))

        false_color_image = create_false_color_image(prediction, mapping)
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
