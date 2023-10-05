from happy.model.imaging_model import ImagingModel
from happy.criteria import Criteria
import numpy as np

from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv3D, Conv2D, MaxPooling3D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate
from keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint


class KerasImageRegressionModel(ImagingModel):
    def __init__(self, data_folder, target, happy_preprocessor=None, additional_meta_data=None, region_selector=None):
        super().__init__(data_folder, target, happy_preprocessor, additional_meta_data, region_selector)
        self.model = None

    def get_y(self, happy_data):
        if happy_data.get_meta_global_data(self.target) is not None:
            return happy_data.get_meta_global_data(self.target)
        else:
            # find any target in image and assume it applies to all...
            crit1 = Criteria("not_missing", key=self.target)
            pixels,_ = happy_data.find_pixels_with_criteria(crit1)
            if not pixels:
                return None
            return happy_data.get_meta_data(x=pixels[0][0], y=pixels[0][1], key=self.target)
        #print("No Y found")
        return None
        
    def fit(self, id_list, target_variable):
        # Implement the training logic for your Keras-based image regression model
        # Use Keras API to create and compile your model, and train it on the data
        # The 'id_list' contains the sample IDs for training, and 'target_variable' is the target for regression

        # Sample code to guide you:

        # Get the training dataset
        training_dataset = self.generate_training_dataset(id_list)

        # Create and compile your Keras image regression model
        model = self.create_keras_image_regression_model()

        # Extract the X_train and y_train from the training dataset
        X_train = np.array(training_dataset["X_train"])
        y_train = np.array(training_dataset["y_train"])
        
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

        # Train the model
        
        learning_rate = 0.05
        adam_optimizer = Adam(learning_rate=learning_rate)
        best_val_loss = float('inf')
        best_model_weights = None

        def save_best_model(epoch, logs):
            nonlocal best_val_loss, best_model_weights
            if logs['val_mae'] < best_val_loss:
                
                best_val_loss = logs['val_mae']
                print("")
                print("using")
                print(best_val_loss)
                best_model_weights = model.get_weights()
                
        best_model_checkpoint = ModelCheckpoint(filepath=None, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        #model.compile(optimizer=adam_optimizer, loss='mse')
        # Compile the model
        model.compile(loss="mse", optimizer=adam_optimizer, metrics=["mse", "mae"])
        model.fit(X_train, y_train, epochs=80, batch_size=8, validation_split=0.2, callbacks=[reduce_lr, keras.callbacks.LambdaCallback(on_epoch_end=save_best_model)])
        #model.fit(X_train, y_train, epochs=10, batch_size=32)
        model.set_weights(best_model_weights)
        # Save the trained model
        self.model = model

    def predict(self, id_list, return_actuals=False):
        # Implement the prediction logic for your Keras-based image regression model
        # Use the loaded Keras model and make predictions on the data
        # The 'id_list' contains the sample IDs for prediction

        # Sample code to guide you:

        # Get the prediction dataset
        prediction_dataset = self.generate_prediction_dataset(id_list)

        # Ensure the model is loaded before making predictions
        #if self.model is None:
        #    self.load_model("trained_model.h5")

        # Extract the X_pred from the prediction dataset
        X_pred = np.array(prediction_dataset["X_pred"])

        # Make predictions using the loaded model
        predictions = self.model.predict(X_pred)

        # Return the predictions
        if return_actuals:
            return predictions, np.array(prediction_dataset["y_pred"])
        else:
            return predictions,None

    def create_keras_image_regression_model(self):
        # Create your Keras image regression model here
        # Use the Keras API to define your model architecture, compile it, and return the model

        # Sample code to guide you:

        # Define the input shape of your image data
        input_shape = self.data_shape #(64, 64, 3)  # Replace with the actual shape of your image data

        # Create the model
        inputs = Input(shape=input_shape)
        x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(1, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def save_model(self, filepath):
        if self.model is not None:
            self.model.save(filepath)

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)

# Your existing 'ImagingModel' class remains unchanged

# ... (rest of the code) ...
