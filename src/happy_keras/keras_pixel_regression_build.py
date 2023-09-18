from PIL import Image
import numpy as np
import argparse
import os

from happy.preprocessors.preprocessors import SpectralNoiseInterpolator, PadPreprocessor, SNVPreprocessor, MultiPreprocessor, DerivativePreprocessor, WavelengthSubsetPreprocessor, StandardScalerPreprocessor
from happy.splitter.happy_splitter import HappySplitter
from happy_keras.model.keras_pixel_regression_model import KerasPixelRegressionModel
from happy.evaluators.regression_evaluator import RegressionEvaluator
from happy.region_extractors.full_region_extractor import FullRegionExtractor


def main():
    parser = argparse.ArgumentParser(description='Evaluate a Keras-based pixel regression model.')
    parser.add_argument('data_folder', type=str, help='Path to the data folder')
    parser.add_argument('target', type=str, help='Name of the target variable')
    parser.add_argument('happy_splitter_file', type=str, help='Path to JSON file containing splits')
    parser.add_argument('output_folder', type=str, help='Path to the output folder')

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    pp = PadPreprocessor(width=128, height=128, pad_value=0)
    subset_indices = list(range(60, 190))
    w = WavelengthSubsetPreprocessor(subset_indices=subset_indices)

    clean = SpectralNoiseInterpolator()
    SNVpp = SNVPreprocessor()
    SGpp = DerivativePreprocessor(window_length=15, deriv=1)
    std = StandardScalerPreprocessor()
    #PCApp = PCAPreprocessor(components=5, percent_pixels=20)
    pp = MultiPreprocessor(preprocessor_list=[w, clean, SNVpp, SGpp, pp])
    
    # Create a FullRegionSelector instance
    region_selector = FullRegionExtractor(region_size=None, target_name=args.target)

    # Create a HappySplitter instance
    happy_splitter = HappySplitter.load_splits_from_json(args.happy_splitter_file)
    train_ids, valid_ids, test_ids = happy_splitter.get_train_validation_test_splits(0, 0)

    # Create a KerasPixelRegressionModel instance
    pixel_regression_model = KerasPixelRegressionModel(data_folder=args.data_folder, target=args.target,region_selector=region_selector, happy_preprocessor=pp)

    # Fit the model
    pixel_regression_model.fit(id_list=train_ids)

    # Predict using the model
    predictions, actuals = pixel_regression_model.predict(id_list=test_ids, return_actuals=True)

    eval = RegressionEvaluator(happy_splitter, pixel_regression_model, args.target)
    eval.accumulate_stats(predictions,actuals,0,0)
    eval.calculate_and_show_metrics()

    max_actual = np.nanmax(actuals)
    min_actual = np.nanmin(actuals)

    # Save the predictions as PNG images
    for i, prediction in enumerate(predictions):
        if np.isnan(min_actual) or np.isnan(max_actual) or min_actual==max_actual:
            print("NaN value detected. Cannot proceed with gradient calculation.")
            continue
        #print(actuals[i])
        false_color_image = create_false_color_image(prediction, min_actual, max_actual)
        false_color_image.save(os.path.join(args.output_folder, f'false_color_{i}.png'))


#def create_prediction_image(prediction):
##    # Create an image from the regression prediction
#    prediction = (prediction * 255).astype(np.uint8)
#    prediction_image = Image.fromarray(prediction)
#    return prediction_image


def create_false_color_image(predictions, min_actual, max_actual):
    # Find the minimum and maximum values of actuals
    predictions = predictions[:,:,0]
  
    # Create an empty array for the false color image
    false_color = np.zeros((predictions.shape[0], predictions.shape[1], 4), dtype=np.uint8)

    max_actual = max_actual * 1.15
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            prediction = predictions[i, j]

            if prediction <= 0:
                # Zero value is transparent               
                #color = [0, 0, 0, 0]
                color = [0, 0, 255, 255]
            elif prediction < min_actual:
                # Values below the minimum are blue
                color = [0, 0, 255, 255]
            elif prediction > max_actual:
                # Values above the maximum are red
                color = [255, 0, 0, 255]
            else:
                # Calculate the gradient color based on the range of actual values
                gradient = (prediction - min_actual) / (max_actual - min_actual)
                r = int(255 * (1 - gradient))
                g = int(255 * (1 - gradient))
                b = int(128 * gradient)
                color = [r, g, b, 255]

            # Assign the color to the false color image
            false_color[i, j] = color

    false_color_image = Image.fromarray(false_color)
    return false_color_image
    

if __name__ == "__main__":
    main()

