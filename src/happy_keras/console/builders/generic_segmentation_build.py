import argparse
import os
import time
import traceback

from happy.base.core import load_class
from happy.evaluators import ClassificationEvaluator
from happy.splitters import HappySplitter
from happy_keras.model.generic import GenericKerasPixelSegmentationModel
from happy_keras.model.segmentation_model import KerasPixelSegmentationModel, create_false_color_image, \
    create_prediction_image


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Build a Keras-based pixel segmentation model using specified class from Python module.',
        prog="happy-generic-keras-segmentation-build",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_folder', type=str, help='Path to the data folder', required=True)
    parser.add_argument('-P', '--python_file', type=str, help='The Python module with the model class to load')
    parser.add_argument('-c', '--python_class', type=str, help='The name of the model class to load')
    parser.add_argument('-t', '--target', type=str, help='Name of the target variable', required=True)
    parser.add_argument('-n', '--num_classes', type=int, default=4, help='The number of classes, used for generating the mapping')
    parser.add_argument('-s', '--happy_splitter_file', type=str, help='Path to JSON file containing splits', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', required=True)

    args = parser.parse_args()

    # there is an optional mapping file in happy data now, but TODO here.
    mapping = {}
    for i in range(args.num_classes):
        mapping[i] = i

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Create a HappySplitter instance
    happy_splitter = HappySplitter.load_splits_from_json(args.happy_splitter_file)
    train_ids, valid_ids, test_ids = happy_splitter.get_train_validation_test_splits(0, 0)

    # create model
    c = load_class(args.python_file, "happy_keras.generic_pixel_regression." + str(int(round(time.time() * 1000))),
                   args.python_class)
    if issubclass(c, KerasPixelSegmentationModel):
        pixel_segmentation_model = GenericKerasPixelSegmentationModel.instantiate(
            c, args.happy_data_base_dir, args.target_value)
    else:
        raise Exception("Unsupported base model class: %s" % str(c))

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
