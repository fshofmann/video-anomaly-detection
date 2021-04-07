from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.io_.batch_generator import BatchGenerator, BatchLoader


class IFTM:
    """Identity Function and Threshold Model (IFTM) by Schmidt et al.
    (https://www.researchgate.net/publication/327484223_IFTM_-_Unsupervised_Anomaly_Detection_for_Virtualized_Network_Function_Services)
    is an anomaly detection approach that utilizes identity functions and forecasting mechanisms to detect anomalies, by
    either reconstructing the current datapoint or forecasting a future one. Then the predicted value is compared to the
    actual data by a reconstruction or prediction error function. To detect anomalies, a threshold is applied to the
    result -- if it exceeds the threshold it is anomaly, else it is not.

    This class is a wrapper of IFTM around (our) video generation models. It was kept generic on purpose. However it
    only implements the non-streaming variant of IFTM, training the threshold model only by cumulative aggregation. This
    is done because we assume the video generator model is trained by batch training and not online, therefore it would
    not make sense to train the TM any other way.

    Does not subclass keras.Model because threshold training is not done in steps or epochs - more than one pass over
    the data would be considered unnecessary (only the sample mean and the sample standard deviation are computed).
    """
    video_generator: keras.Model
    error_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    threshold: int

    def __init__(self, video_generator: keras.Model, error_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]):
        """Inits a new IFTM with a given TRAINED Keras model and a custom error function. Sets the threshold to -1,
        i.e. all data points are predicted to be anomalies.

        :param video_generator: Trained video generator network.
        :param error_function: Function that computes the error between predicted and actual video for an entire batch,
        for each sample individually.
        """
        self.video_generator = video_generator
        self.error_function = error_function
        self.threshold = -1

    def train_threshold(self, training_batches: BatchGenerator, max_queue_size=16, no_workers: int = 4) -> None:
        """Run the entire training data provided on the trained video generator model and predict the training data's
        error with the passed error function. The results are then used to compute the threshold model (using cumulative
        aggregation).

        :param training_batches: BatchGenerator for data that was used to train the video generator model.
        :param max_queue_size: Size of cache for loaded batches during training.
        :param no_workers: Number of threads to load batches from generator during training.
        """
        batch_loader: BatchLoader = BatchLoader(training_batches, max_queue_size, no_workers)
        predictions = []
        for _ in range(len(batch_loader)):
            input_videos, actual_videos = batch_loader.get_batch()
            generated_videos = self.video_generator(input_videos, training=False)
            identity_results = self.error_function(generated_videos, tf.convert_to_tensor(actual_videos))
            predictions.append(identity_results)
        batch_loader.shutdown_workers()

        predictions = np.array(predictions)
        mean = np.mean(predictions)
        std = np.std(predictions, ddof=1)

        self.threshold = mean + std

    def predict(self, input_videos: np.ndarray, actual_videos: np.ndarray) -> (np.ndarray, np.ndarray):
        """Do video anomaly detection on the given video batch. Will classify all samples as anomalies if threshold was
        not trained.

        :param input_videos: Input video clips as array (samples, frames, height, width, channels).
        :param actual_videos: Actual video clips as array (samples, frames, height, width, channels).
        :return:
            - identity_results - Identity function values for each video sample.
            - predictions      - Labels for each video sample (True=Anomaly).
        """
        generated_videos = self.video_generator(input_videos, training=False)
        identity_results = self.error_function(generated_videos, tf.convert_to_tensor(actual_videos)).numpy()
        predictions = identity_results > self.threshold

        return identity_results, predictions
