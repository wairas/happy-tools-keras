import argparse
import logging
import os
import traceback

from wai.logging import add_logging_level, set_logging_level
from happy.base.app import init_app
from happy.preprocessors import Preprocessor, MultiPreprocessor
from happy.region_extractors import FullRegionExtractor
from happy.splitters import DataSplits
from happy_keras.models.unsupervised_segmentation import KerasUnsupervisedSegmentationModel, \
    create_prediction_image, create_false_color_image


PROG = "happy-keras-unsupervised-build"

logger = logging.getLogger(PROG)


def default_preprocessors() -> str:
    args = [
        "wavelength-subset -f 60 -t 189",
        "snv",
        "derivative",
        "pad -W 128 -H 128 -v 0",
    ]
    return " ".join(args)


def main():
    # Parse command-line arguments
    init_app()
    parser = argparse.ArgumentParser(
        description='Build a Keras-based unsupervised segmentation model.',
        prog=PROG,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_folder', type=str, help='Path to the data folder', required=True)
    parser.add_argument('-P', '--preprocessors', type=str, help='The preprocessors to apply to the data. Either preprocessor command-line(s) or file with one preprocessor command-line per line.', required=False, default=default_preprocessors())
    parser.add_argument('-t', '--target', type=str, help='Name of the target variable', required=True)
    parser.add_argument('-n', '--num_clusters', type=int, default=4, help='The number of clusters to use')
    parser.add_argument('-s', '--splits_file', type=str, help='Path to JSON file containing splits', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', required=True)
    add_logging_level(parser, short_opt="-V")

    args = parser.parse_args()
    set_logging_level(logger, args.logging_level)

    # preprocessors
    logger.info("Creating pre-processing")
    preproc = MultiPreprocessor(preprocessor_list=Preprocessor.parse_preprocessors(args.preprocessors))

    # Create the output folder if it doesn't exist
    logger.info("Creating output dir: %s" % args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)

    # Create a FullRegionSelector instance
    logger.info("Creating region extractor")
    region_selector = FullRegionExtractor(region_size=None, target_name=args.target)

    # splits
    logger.info("Loading splits: %s" % args.splits_file)
    splits = DataSplits.load(args.splits_file)
    train_ids, valid_ids, test_ids = splits.get_train_validation_test_splits(0, 0)
    
    # Create a KerasUnsupervisedSegmentationModel instance
    logger.info("Creating model")
    unsupervised_segmentation_model = KerasUnsupervisedSegmentationModel(
        data_folder=args.data_folder, target=args.target, num_clusters=args.num_clusters,
        region_selector=region_selector, happy_preprocessor=preproc)

    # Fit the model
    logger.info("Fitting model...")
    unsupervised_segmentation_model.fit(id_list=train_ids, target_variable=args.target)
    
    # Predict using the model
    logger.info("Predicting...")
    predictions, _ = unsupervised_segmentation_model.predict(id_list=test_ids, return_actuals=False)

    # Save the predictions as PNG images
    for i, prediction in enumerate(predictions):
        prediction_image = create_prediction_image(prediction)
        prediction_image.save(os.path.join(args.output_folder, f'prediction_{i}.png'))

        false_color_image = create_false_color_image(prediction, args.num_clusters)
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
