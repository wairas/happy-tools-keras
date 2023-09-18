import os
import numpy as np
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, GlobalAveragePooling2D, Dense, Reshape, Flatten
from tensorflow import keras
from happy.model.imaging_model import ImagingModel
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50


class KerasPixelRegressionModel(ImagingModel):
    def __init__(self, data_folder, target, happy_preprocessor=None, additional_meta_data=None, region_selector=None, mapping=None):
        super().__init__(data_folder, target, happy_preprocessor, additional_meta_data, region_selector)
        self.model = None
        self.mapping = mapping
        self.training_data = None
        
    def get_y(self, happy_data):
        raw_y = happy_data.get_meta_data(key=self.target)
        return raw_y
        
    def fit(self, id_list, keep_training_data=False):
        training_dataset = self.generate_training_dataset(id_list)

        model = self.create_keras_pixel_regression_model()#create_keras_pixel_regression_model()
        
        X_train = np.array(training_dataset["X_train"])
        y_train = np.array(training_dataset["y_train"])
        
        if keep_training_data:
            self.training_data = training_dataset
      
        learning_rate = 0.01
        adam_optimizer = Adam(learning_rate=learning_rate)

        model.compile(loss="mean_squared_error", optimizer=adam_optimizer, metrics=["mean_absolute_error"])

        # Add ReduceLROnPlateau callback
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0001)

        model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[reduce_lr])

        self.model = model

    def predict(self, id_list, return_actuals=False):
        prediction_dataset = self.generate_prediction_dataset(id_list, return_actuals=return_actuals)
        X_pred = np.array(prediction_dataset["X_pred"])
        predictions = self.model.predict(X_pred)
        if return_actuals:
            return predictions, np.array(prediction_dataset["y_pred"])
        else:
            return predictions, None

    def create_keras_pixel_regression_model(self):
        input_shape = self.get_data_shape() #self.data_shape

        # if the input size is not a multiple of 4, this is going to fail...
        inputs = Input(shape=input_shape)
        x = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = Conv2D(1, kernel_size=(1, 1), activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=x)
        model.summary()
        return model

    def create_keras_pixel_regression_model_resnet(self):
        input_shape = self.get_data_shape()  # Assuming (height, width, channels) input shape
        print(input_shape)
        # Create a CNN layer to learn features from the input data
        inputs = Input(shape=input_shape)
        cnn_features = Conv2D(3, kernel_size=(1, 1), activation="relu", padding="same")(inputs)
        
        # Load pre-trained ResNet-50 model (excluding top layers)
        resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=cnn_features.shape[1:])
        
        # Freeze the layers of the ResNet-50 model
        #for layer in resnet_base.layers:
        #    layer.trainable = False
        
        # Create a model with CNN features followed by ResNet-50 features
        resnet_features = resnet_base(cnn_features)
        
        # Add regression layers on top of combined features
        x = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")(resnet_features)
        x = Flatten()(x)
        # ... add more layers as needed
        # Calculate the expected dimensions of the flattened ResNet output
        expected_flatten_dim = input_shape[0] * input_shape[1]# * input_shape[2]
        print(expected_flatten_dim)

        # Add a Dense layer with the expected_flatten_dim units
        x = Dense(expected_flatten_dim, activation="relu")(x)

        # Reshape the dense layer output to match the dimensions of the target shape
        reshaped_output = Reshape((input_shape[0], input_shape[1], 1))(x)
        
        # Output layer with same dimensions as input
        #output = Conv2D(input_shape[-1], kernel_size=(1, 1), activation="linear", padding="same")(x)
        
        model = keras.Model(inputs=inputs, outputs=reshaped_output)
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
