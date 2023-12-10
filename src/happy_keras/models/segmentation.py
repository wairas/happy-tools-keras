import os
import numpy as np
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow import keras
from happy.models.imaging import ImagingModel


class KerasPixelSegmentationModel(ImagingModel):
    def __init__(self, data_folder, target, happy_preprocessor=None, additional_meta_data=None, region_selector=None, mapping=None):
        super().__init__(data_folder, target, happy_preprocessor, additional_meta_data, region_selector)
        self.model = None
        self.mapping = mapping
        
        # Extract values from the dictionary
        values = mapping.values()
        # Convert values to a list and get unique values
        unique_values = set(values)
        # Count the number of unique values
        self.num_classes = len(unique_values)
    
    """
    def get_y(self, happy_data):
        raw_y = happy_data.get_meta_data(key=self.target)
        # now turn into one-hot
        print(f"raw_y-shape: {raw_y.shape}")
        one_hot = np.eye(self.num_classes)[raw_y]
       
        print(f"one_hot-shape: {one_hot.shape}")
        # You may need to adjust this to obtain the pixel-level labels from your data
        return one_hot#, self.mapping)
    """
    def get_y(self, happy_data):
        
        raw_y = happy_data.get_meta_data(key=self.target)
        
        # Determine the dimensions of raw_y
        height, width = raw_y.shape[0], raw_y.shape[1]
        
        # Turn raw_y into one-hot encoding
        one_hot = np.eye(self.num_classes)[raw_y.reshape(-1)]
        one_hot = one_hot.reshape((height, width, self.num_classes))
        
        self.logger().info(f"raw_y-shape: {raw_y.shape}")
        self.logger().info(f"one_hot-shape: {one_hot.shape}")
        return one_hot
        
    def fit(self, id_list, target_variable):
        training_dataset = self.generate_training_dataset(id_list)

        model = self.create_keras_pixel_segmentation_model_concat()

        # filtered_arrays_X = [arr for arr in training_dataset["X_train"] if len(arr.shape) >= 2 and arr.shape[:2] == (128, 128)]
        # filtered_arrays_y = [arr for arr in training_dataset["y_train"] if len(arr.shape) >= 2 and arr.shape[:2] == (128, 128)]
        
        X_train = np.array(training_dataset["X_train"])
        y_train = np.array(training_dataset["y_train"])

        #  X_train = np.array(filtered_arrays_X)
        #  y_train = np.array(filtered_arrays_y)

        for i in X_train:
            self.logger().info(i.shape)
            
        for i in y_train:
            self.logger().info(i.shape)

        self.logger().info("X_train shape:", X_train.shape)
        self.logger().info("y_train shape:", y_train.shape)

        learning_rate = 0.001
        adam_optimizer = Adam(learning_rate=learning_rate)

        model.compile(loss="categorical_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"])

        model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

        self.model = model

    def predict(self, id_list, return_actuals=False):
        prediction_dataset = self.generate_prediction_dataset(id_list, return_actuals=return_actuals)
        X_pred = np.array(prediction_dataset["X_pred"])
        predictions = self.model.predict(X_pred)
        if return_actuals:
            return predictions, np.array(prediction_dataset["y_pred"])
        else:
            return predictions,None

    def create_keras_pixel_segmentation_model_concat(self):
        input_shape = self.get_data_shape() # self.data_shape

        # if the input size is not a multiple of 4, this is going to fail...
        inputs = Input(shape=input_shape)
        
        # Encoder
        conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(pool2)
        
        # Decoder
        up1 = UpSampling2D(size=(2, 2))(conv3)
        concat1 = concatenate([up1, conv2], axis=-1)
        conv4 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(concat1)
        up2 = UpSampling2D(size=(2, 2))(conv4)
        concat2 = concatenate([up2, conv1], axis=-1)
        conv5 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(concat2)
        
        # Output layer
        output = Conv2D(self.num_classes, kernel_size=(1, 1), activation="softmax")(conv5)

        model = keras.Model(inputs=inputs, outputs=output)
        model.summary()
        return model
        
    def create_keras_pixel_segmentation_model(self):
        input_shape = self.get_data_shape() #self.data_shape

        # if the input size is not a multiple of 4, this is going to fail...
        inputs = Input(shape=input_shape)
        x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = Conv2D(self.num_classes, kernel_size=(1, 1), activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=x)
        model.summary()
        return model

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Save model architecture
        model_architecture_path = os.path.join(folder, 'model_architecture.json')
        with open(model_architecture_path, 'w') as f:
            f.write(self.model.to_json())

        # Save model weights
        model_weights_path = os.path.join(folder, 'model_weights.h5')
        self.model.save_weights(model_weights_path)

        # Save class attributes
        attributes = {
            'data_folder': self.data_folder,
            'target': self.target,
            'happy_preprocessor': self.happy_preprocessor,
            'additional_meta_data': self.additional_meta_data,
            'region_selector': self.region_selector,
            'mapping': self.mapping
        }
        attributes_path = os.path.join(folder, 'attributes.npy')
        np.save(attributes_path, attributes)
        
        # Generate Python code to instantiate the model
        model_code = generate_model_code(model, config_filepath)

        # Save the model code to a Python file
        model_filename = os.path.join(folder, 'generated_model.py')
        with open(model_filename, 'w') as model_file:
            model_file.write(model_code)

    @classmethod
    def load(cls, folder):
        attributes_path = os.path.join(folder, 'attributes.npy')
        attributes = np.load(attributes_path, allow_pickle=True).item()

        instance = cls(**attributes)
        
        # Load model architecture
        model_architecture_path = os.path.join(folder, 'model_architecture.json')
        with open(model_architecture_path, 'r') as f:
            model_json = f.read()
            instance.model = keras.models.model_from_json(model_json)

        # Load model weights
        model_weights_path = os.path.join(folder, 'model_weights.h5')
        instance.model.load_weights(model_weights_path)

        return instance
        
    def generate_model_code(model, config_filepath):
        model_code = f"""
import os
from model.keras_pixel_segmentation_model import KerasPixelSegmentationModel

# Load model parameters from the config file
with open('{config_filepath}', 'r') as config_file:
    model_config = json.load(config_file)

# Instantiate the model
model = KerasPixelSegmentationModel(**model_config)

# Load the trained model weights
model.load_model('{model.model_folder}')

# Now you can use the 'model' object for predictions or other tasks
    """

        return model_code
