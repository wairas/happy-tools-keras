import argparse
import logging
import os
import time
import traceback

from wai.logging import add_logging_level, set_logging_level
from happy.base.core import load_class
from happy.base.app import init_app
from happy.splitters import HappySplitter
from happy_keras.models.generic import GenericKerasUnsupervisedSegmentationModel
from happy_keras.models.unsupervised_segmentation import KerasUnsupervisedSegmentationModel, \
    create_prediction_image, create_false_color_image


PROG = "happy-generic-keras-unsupervised-build"

logger = logging.getLogger(PROG)


def main():
    # Parse command-line arguments
    init_app()
    parser = argparse.ArgumentParser(
        description='Build a Keras-based unsuperivised model using specified class from Python module.',
        prog=PROG,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_folder', type=str, help='Path to the data folder', required=True)
    parser.add_argument('-P', '--python_file', type=str, help='The Python module with the model class to load')
    parser.add_argument('-c', '--python_class', type=str, help='The name of the model class to load')
    parser.add_argument('-t', '--target', type=str, help='Name of the target variable', required=True)
    parser.add_argument('-n', '--num_clusters', type=int, default=4, help='The number of clusters to use')
    parser.add_argument('-s', '--happy_splitter_file', type=str, help='Path to JSON file containing splits', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', required=True)
    add_logging_level(parser, short_opt="-V")

    args = parser.parse_args()
    set_logging_level(logger, args.logging_level)

    # Create the output folder if it doesn't exist
    logger.info("Creating output dir: %s" % args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)

    # Create a HappySplitter instance
    logger.info("Loading splits: %s" % args.happy_splitter_file)
    happy_splitter = HappySplitter.load_splits_from_json(args.happy_splitter_file)
    train_ids, valid_ids, test_ids = happy_splitter.get_train_validation_test_splits(0, 0)

    # create model
    logger.info("Loading class %s from: %s" % (args.python_class, args.python_file))
    c = load_class(args.python_file, "happy_keras.generic_pixel_regression." + str(int(round(time.time() * 1000))),
                   args.python_class)
    if issubclass(c, KerasUnsupervisedSegmentationModel):
        unsupervised_segmentation_model = GenericKerasUnsupervisedSegmentationModel.instantiate(
            c, args.happy_data_base_dir, args.target_value)
    else:
        raise Exception("Unsupported base model class: %s" % str(c))

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
