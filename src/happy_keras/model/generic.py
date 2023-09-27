from happy_keras.model.pixel_regression_model import KerasPixelRegressionModel
from happy_keras.model.segmentation_model import KerasPixelSegmentationModel
from happy_keras.model.unsupervised_segmentation_model import KerasUnsupervisedSegmentationModel


class GenericKerasPixelRegressionModel(KerasPixelRegressionModel):

    def __init__(self, base_model):
        super().__init__(self, None, None)
        self.base_model = base_model

    def generate_batch(self, sample_ids, batch_size, is_train=True, loop=False, return_actuals=False):
        for item in self.base_model.generate_batch(sample_ids, batch_size, is_train=is_train, loop=loop, return_actuals=return_actuals):
            yield item

    def generate_training_dataset(self, sample_ids):
        return self.base_model.generate_training_dataset(sample_ids)

    def generate_prediction_dataset(self, sample_ids, pixel_selector=None, return_actuals=False):
        return self.base_model.generate_prediction_dataset(sample_ids, pixel_selector=pixel_selector, return_actuals=return_actuals)

    def set_region_selector(self, region_selector):
        self.base_model.set_region_selector(region_selector)

    def preprocess_data(self, happy_data):
        return self.base_model.preprocess_data(happy_data)

    def get_data_shape(self):
        return self.base_model.get_data_shape()

    def get_y(self, happy_data):
        return self.base_model.get_y(happy_data)

    def fit(self, id_list, keep_training_data=False):
        self.base_model.fit(id_list, keep_training_data=keep_training_data)

    def predict(self, id_list, return_actuals=False):
        return self.base_model.predict(id_list, return_actuals=return_actuals)

    @classmethod
    def instantiate(cls, c, data_folder, target):
        if not issubclass(c, KerasPixelRegressionModel):
            raise Exception("Class '%s' not of type '%s'!" % (str(c), str(KerasPixelRegressionModel)))
        base_model = c(data_folder, target)
        return GenericKerasPixelRegressionModel(base_model)


class GenericKerasPixelSegmentationModel(KerasPixelSegmentationModel):

    def __init__(self, base_model):
        super().__init__(self, None, None)
        self.base_model = base_model

    def get_data_shape(self):
        return self.base_model.get_data_shape()

    def generate_batch(self, sample_ids, batch_size, is_train=True, loop=False, return_actuals=False):
        for item in self.base_model.generate_batch(sample_ids, batch_size, is_train=is_train, loop=loop, return_actuals=return_actuals):
            yield item

    def generate_training_dataset(self, sample_ids):
        return self.base_model.generate_training_dataset(sample_ids)

    def generate_prediction_dataset(self, sample_ids, pixel_selector=None, return_actuals=False):
        return self.base_model.generate_prediction_dataset(sample_ids, pixel_selector=pixel_selector, return_actuals=return_actuals)

    def set_region_selector(self, region_selector):
        self.base_model.set_region_selector(region_selector)

    def get_y(self, happy_data):
        return self.base_model.get_y(happy_data)

    def fit(self, id_list, target_variable):
        self.base_model.fit(id_list, target_variable)

    def predict(self, id_list, return_actuals=False):
        return self.base_model.predict(id_list, return_actuals=return_actuals)

    def create_keras_pixel_segmentation_model_concat(self):
        return self.base_model.create_keras_pixel_segmentation_model_concat()

    def create_keras_pixel_segmentation_model(self):
        return self.base_model.create_keras_pixel_segmentation_model()

    def save(self, folder):
        self.base_model.save(folder)

    @classmethod
    def instantiate(cls, c, data_folder, target):
        if not issubclass(c, KerasPixelSegmentationModel):
            raise Exception("Class '%s' not of type '%s'!" % (str(c), str(KerasPixelSegmentationModel)))
        base_model = c(data_folder, target)
        return GenericKerasPixelSegmentationModel(base_model)


class GenericKerasUnsupervisedSegmentationModel(KerasUnsupervisedSegmentationModel):

    def __init__(self, base_model):
        super().__init__(self, None, None)
        self.base_model = base_model

    def get_data_shape(self):
        return self.base_model.get_data_shape()

    def get_y(self, happy_data):
        return self.base_model.get_y(happy_data)

    def generate_batch(self, sample_ids, batch_size, is_train=True, loop=False, return_actuals=False):
        for item in self.base_model.generate_batch(sample_ids, batch_size, is_train=is_train, loop=loop, return_actuals=return_actuals):
            yield item

    def generate_training_dataset(self, sample_ids):
        return self.base_model.generate_training_dataset(sample_ids)

    def generate_prediction_dataset(self, sample_ids, pixel_selector=None, return_actuals=False):
        return self.base_model.generate_prediction_dataset(sample_ids, pixel_selector=pixel_selector, return_actuals=return_actuals)

    def set_region_selector(self, region_selector):
        self.base_model.set_region_selector(region_selector)

    def fit(self, id_list, target_variable):
        self.base_model.fit(id_list, target_variable)

    def predict(self, id_list, return_actuals=False):
        return self.base_model.predict(id_list, return_actuals=return_actuals)

    def channel_attention(self, input_tensor):
        return self.base_model.channel_attention(input_tensor)

    def create_autoencoder_segmentation_model_bands(self):
        return self.base_model.create_autoencoder_segmentation_model_bands()

    def create_autoencoder_segmentation_model(self):
        return self.base_model.create_autoencoder_segmentation_model()

    @classmethod
    def instantiate(cls, c, data_folder, target):
        if not issubclass(c, KerasUnsupervisedSegmentationModel):
            raise Exception("Class '%s' not of type '%s'!" % (str(c), str(KerasUnsupervisedSegmentationModel)))
        base_model = c(data_folder, target)
        return GenericKerasUnsupervisedSegmentationModel(base_model)
