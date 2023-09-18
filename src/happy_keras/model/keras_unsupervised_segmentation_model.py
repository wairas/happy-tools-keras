import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Dense, Flatten, Multiply, Lambda
from tensorflow import keras
from happy.model.imaging_model import ImagingModel
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.constraints import Constraint


class SingleNonZeroConstraint(Constraint):
    def __call__(self, w):
        normalized_w = w / tf.reduce_sum(w, axis=-1, keepdims=True)
        return w.assign(normalized_w)


class SpatialRegularizedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.01, beta=0.001, **kwargs):
        super(SpatialRegularizedLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        
    def call(self, y_true, y_pred):
        # Calculate the standard Mean Squared Error (MSE) loss
        mse_loss = MeanSquaredError()(y_true, y_pred)
        
        # Calculate the spatial regularization term using total_variation
        spatial_regularization = self.beta * tf.reduce_sum(tf.image.total_variation(y_pred))
        
        # Combine the MSE loss and the spatial regularization term
        total_loss = mse_loss + spatial_regularization
        
        return self.alpha * total_loss


class ContrastiveDivergenceLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.1, **kwargs):
        super(ContrastiveDivergenceLoss, self).__init__(**kwargs)
        self.alpha = alpha
        
    def call(self, y_true, y_pred):
        loss = tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(1 - y_pred, 0)))
        return self.alpha * loss


class KerasUnsupervisedSegmentationModel(ImagingModel):
    def __init__(self, data_folder, target, num_clusters, happy_preprocessor=None, additional_meta_data=None, region_selector=None, mapping=None):
        super().__init__(data_folder, target, happy_preprocessor, additional_meta_data, region_selector)
        self.model = None
        self.mapping = mapping
        self.num_clusters = num_clusters
        self.predict_model = None
    
    def fit(self, id_list, target_variable):
        # for unsupervised, there is no Y, so just use prediction data
        training_dataset = self.generate_prediction_dataset(id_list)

        model = self.create_autoencoder_segmentation_model_bands()
        
        X_train = np.array(training_dataset["X_pred"])
        
        learning_rate = 0.01
        adam_optimizer = Adam(learning_rate=learning_rate)

        #model.compile(loss=ContrastiveDivergenceLoss(alpha=0.1), optimizer=adam_optimizer)
        model.compile(optimizer=adam_optimizer, loss=MeanSquaredError())# loss=SpatialRegularizedLoss())  #  # Set the loss function here

        model.fit(X_train, X_train, epochs=70, batch_size=8, validation_split=0.2)

        self.model = model
        
    def predict(self, id_list, return_actuals=False):
        prediction_dataset = self.generate_prediction_dataset(id_list, return_actuals=return_actuals)
        X_pred = np.array(prediction_dataset["X_pred"])
        
        predictions = self.predict_model.predict(X_pred)
        if return_actuals:
            return predictions, np.array(prediction_dataset["y_pred"])
        else:
            return predictions, None

    def channel_attention(self, input_tensor):
        # Compute channel attention across all channels
        channel_attention = Conv2D(input_tensor.shape[-1], (1, 1), activation='sigmoid', padding='same')(input_tensor)
        
        attention_result = Multiply()([input_tensor, channel_attention])
    
        return attention_result

    def create_autoencoder_segmentation_model_bands(self):
        input_shape = self.get_data_shape()  # self.data_shape
        
        # Encoder
        encoder_input = Input(shape=input_shape)

        # Apply channel-wise attention
        x = self.channel_attention(encoder_input)

        # Parallel Reduced-Channel CNNs
        num_channels = input_shape[-1]
        band_experts = []
        for i in range(num_channels):
            band_input = Lambda(lambda x: x[:, :, :, i:i+1])(x)
            cnn = Conv2D(8, kernel_size=(3, 3), activation="relu", padding="same")(band_input)
            pool1 = MaxPooling2D(pool_size=(2, 2))(cnn)
            band_experts.append(pool1)

        # Concatenate outputs of parallel CNNs
        combined_features = concatenate([band_expert for band_expert in band_experts], axis=-1)
        combined_features = UpSampling2D(size=(2, 2))(combined_features)
         
        # Bottleneck
        bottleneck = Conv2D(self.num_clusters, kernel_size=(1, 1), activation='softmax', padding="same")(combined_features)

        # Decoder
        x = Conv2D(8, kernel_size=(3, 3), activation="relu", padding="same")(bottleneck)
        x = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")(x)
        decoded = Conv2D(input_shape[-1], kernel_size=(3, 3), activation="linear", padding="same")(x)

        model = Model(inputs=encoder_input, outputs=decoded)
        model.summary()

        self.predict_model = Model(encoder_input, bottleneck)
        return model

    def create_autoencoder_segmentation_model(self):
        input_shape = self.get_data_shape()  # self.data_shape
        
        # Encoder
        encoder_input = Input(shape=input_shape)
        """
        x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(encoder_input)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(x)
        bottleneck = Conv2D(self.num_clusters, kernel_size=(1, 1), activation='softmax', padding="same")(x)

        # Decoder
        x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(bottleneck)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(x)
        #decoded = Conv2D(input_shape[-1], kernel_size=(3, 3), activation="sigmoid", padding="same")(x)
        decoded = Conv2D(input_shape[-1], kernel_size=(3, 3), activation="linear", padding="same")(x)
        """
        x = self.channel_attention(encoder_input)  # Apply channel-wise attention
        #x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(encoder_input)
        x = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")(x)
        
        x = Conv2D(8, kernel_size=(3, 3), activation="relu", padding="same")(x)
        bottleneck = Conv2D(self.num_clusters, kernel_size=(1, 1), activation='softmax', padding="same")(x) #, (x) #, kernel_constraint=SingleNonZeroConstraint())(x)

        # Decoder
        x = Conv2D(8, kernel_size=(3, 3), activation="relu", padding="same")(bottleneck)
        x = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")(x)
        #x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(x)
        #decoded = Conv2D(input_shape[-1], kernel_size=(3, 3), activation="sigmoid", padding="same")(x)
        decoded = Conv2D(input_shape[-1], kernel_size=(3, 3), activation="linear", padding="same")(x)
        
        model = keras.Model(inputs=encoder_input, outputs=decoded)
        model.summary()
        
        self.predict_model = keras.Model(encoder_input, bottleneck)
        return model

    # Other methods (save, load, etc.) can remain unchanged or be adapted as needed
