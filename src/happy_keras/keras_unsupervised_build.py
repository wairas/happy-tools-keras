from PIL import Image
from matplotlib import cm

import os
import numpy as np

import argparse
from happy.splitter.happy_splitter import HappySplitter

from happy_keras.model.keras_unsupervised_segmentation_model import KerasUnsupervisedSegmentationModel
from happy.region_extractors.full_region_extractor import FullRegionExtractor
from happy.preprocessors.preprocessors import PadPreprocessor, PCAPreprocessor, SNVPreprocessor, MultiPreprocessor, DerivativePreprocessor, WavelengthSubsetPreprocessor


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Build a Keras-based pixel segmentation model.')
    parser.add_argument('data_folder', type=str, help='Path to the data folder')
    parser.add_argument('target', type=str, help='Name of the target variable')
    parser.add_argument('happy_splitter_file', type=str, help='Path to JSON file containing splits')
    parser.add_argument('output_folder', type=str, help='Path to the output folder')

    args = parser.parse_args()
    
    # Define the number of clusters
    num_clusters = 4  # You can adjust this based on your data
    
    pp = PadPreprocessor(width=128, height=128, pad_value=0)
    subset_indices = list(range(60, 190))
    w = WavelengthSubsetPreprocessor(subset_indices=subset_indices)

    SNVpp = SNVPreprocessor()
    SGpp = DerivativePreprocessor()
    PCApp = PCAPreprocessor(components=5, percent_pixels=20)
    pp = MultiPreprocessor(preprocessor_list=[w, SNVpp, SGpp, PCApp, pp])
    
    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Create a FullRegionSelector instance
    region_selector = FullRegionExtractor(region_size=None, target_name=args.target)
    happy_splitter = HappySplitter.load_splits_from_json(args.happy_splitter_file)
    train_ids, valid_ids, test_ids = happy_splitter.get_train_validation_test_splits(0, 0)
    
    # Create a KerasUnsupervisedSegmentationModel instance
    unsupervised_segmentation_model = KerasUnsupervisedSegmentationModel(data_folder=args.data_folder, target=args.target, num_clusters=num_clusters, region_selector=region_selector, happy_preprocessor=pp)

    # Load sample IDs (you can modify this based on your folder structure)
    # sample_ids = [f.name for f in os.scandir(args.data_folder) if f.is_dir()]

    # Fit the model
    unsupervised_segmentation_model.fit(id_list=train_ids, target_variable=args.target)
    
    # Predict using the model
    predictions, _ = unsupervised_segmentation_model.predict(id_list=test_ids, return_actuals=False)
    #eval = ClassificationEvaluator(happy_splitter, unsupervised_segmentation_model, args.target)
    #eval.accumulate_stats(predictions, actuals, 0, 0)
    #eval.calculate_and_show_metrics()

    # Save the predictions as PNG images
    for i, prediction in enumerate(predictions):
        prediction_image = create_prediction_image(prediction)
        prediction_image.save(os.path.join(args.output_folder, f'prediction_{i}.png'))

        false_color_image = create_false_color_image(prediction, num_clusters)
        false_color_image.save(os.path.join(args.output_folder, f'false_color_{i}.png'))


def create_prediction_image(prediction):
    # Create a grayscale prediction image
    prediction = np.argmax(prediction, axis=-1)
    prediction_image = Image.fromarray(prediction.astype(np.uint8))
    return prediction_image


def create_false_color_image(prediction, num_clusters):
    # Create a false color prediction image
    prediction = np.argmax(prediction, axis=-1)
    cmap = cm.get_cmap('viridis', num_clusters)
    false_color = cmap(prediction)
    false_color_image = Image.fromarray((false_color[:, :, :3] * 255).astype(np.uint8))
    return false_color_image


if __name__ == "__main__":
    main()
