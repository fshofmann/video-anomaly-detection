from typing import Callable, Mapping

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Optimizer


# noinspection PyAbstractClass
class VGAN(keras.Model):
    """Model wrapper of the VGAN framework by Vondrick et al. (http://www.cs.columbia.edu/~vondrick/tinyvideo/). Kept
    generic on purpose to support different kinds of configurations, loss functions and optimizers. In actuality, this
    class is a normal GAN. The code in its original form appeared in the `vgan` notebook found in `src/models`.

    """
    generator: keras.Model
    discriminator: keras.Model
    g_optimizer: Optimizer
    d_optimizer: Optimizer
    g_loss_fn: Callable[[tf.Tensor], tf.Tensor]
    d_loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    g_loss_tracker: keras.metrics.Metric
    d_loss_tracker: keras.metrics.Metric

    def __init__(self, generator: keras.Model, discriminator: keras.Model, latent_dim: int):
        """Creates a new VGAN model, that consists of two adversarial networks.

        :param generator: Generator network.
        :param discriminator: Discriminator network.
        :param latent_dim: Dimensions of latent input space.
        """
        super(VGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        """Calls the generator model on a set of inputs.

        :param inputs: Input for generator.
        :param training: Whether network should be run in training mode or inference mode.
        :param mask: Tensor mask; not implemented and therefore discarded.
        :return: Output of generator (generated videos).
        """
        return self.generator(inputs, training=training)

    # noinspection PyMethodOverriding
    def compile(self, g_optimizer: Optimizer, d_optimizer: Optimizer,
                g_loss_fn: Callable[[tf.Tensor], tf.Tensor], d_loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]):
        """Configures the models for training.

        THIS METHOD HAS A DIFFERENT SIGNATURE THAN THE OVERRIDDEN METHOD! DO NOT USE THIS MODEL IN ITS ORIGINAL CONTEXT!

        :param g_optimizer: Generator optimizer instance.
        :param d_optimizer: Discriminator optimizer instance.
        :param g_loss_fn: Function that computes the generator loss.
        :param d_loss_fn: Function that computes the discriminator loss.
        """
        super(VGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")

    def train_step(self, data) -> Mapping[str, float]:
        """Computes a single train step, i.e. training over one batch.

        :param data: Data tuple of a single batch.
            - input_videos - Beginning/Snippet of actual video clips. Discarded for conventional VGAN.
            - real_videos  - Actual video clips. Input for discriminator.
        :return: Dictionary of metrics.
            - g_loss - Averaged generator loss for current epoch.
            - d_loss - Averaged discriminator loss for current epoch.
        """
        _, real_videos = data
        batch_size = tf.shape(real_videos)[0]
        noise_inputs = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_videos = self.generator(noise_inputs, training=True)

            real_output = self.discriminator(real_videos, training=True)
            fake_output = self.discriminator(generated_videos, training=True)

            g_loss = self.g_loss_fn(fake_output)
            d_loss = self.d_loss_fn(real_output, fake_output)

        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {"g_loss": self.g_loss_tracker.result(), "d_loss": self.d_loss_tracker.result()}

    def test_step(self, data) -> Mapping[str, float]:
        """Computes a single evaluation step, i.e. evaluation over one batch.

        :param data: Data tuple of a single batch.
            - input_videos - Beginning/Snippet of actual video clips. Discarded for conventional VGAN.
            - real_videos  - Actual video clips. Input for discriminator.
        :return: Dictionary of metrics.
            - g_loss - Averaged generator loss for current epoch.
            - d_loss - Averaged discriminator loss for current epoch.
        """
        _, real_videos = data
        batch_size = tf.shape(real_videos)[0]
        noise_inputs = tf.random.normal([batch_size, self.latent_dim])

        generated_videos = self.generator(noise_inputs, training=False)

        real_output = self.discriminator(real_videos, training=False)
        fake_output = self.discriminator(generated_videos, training=False)

        g_loss = self.g_loss_fn(fake_output)
        d_loss = self.d_loss_fn(real_output, fake_output)

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {"g_loss": self.g_loss_tracker.result(), "d_loss": self.d_loss_tracker.result()}

    @property
    def metrics(self):
        """Returns the model's metrics added using `compile`. Metrics are reset every epoch."""
        return [self.g_loss_tracker, self.d_loss_tracker]


# noinspection PyAbstractClass
class VGANConditional(keras.Model):
    """Model wrapper of the VGAN framework by Vondrick et al. (http://www.cs.columbia.edu/~vondrick/tinyvideo/),
    extended for conditional video generation (C-VGAN). Kept generic on purpose to support different kinds of
    configurations, loss functions and optimizers, therefore allows the use of the original C-VGAN approach in addition
    to our proposed forecasting model. The code in its original form appeared in the `vgan_conditional` notebooks found
    in `src/models`.

    """
    generator: keras.Model
    discriminator: keras.Model
    g_optimizer: Optimizer
    d_optimizer: Optimizer
    g_loss_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]
    d_loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    g_loss_tracker: keras.metrics.Metric
    d_loss_tracker: keras.metrics.Metric

    def __init__(self, generator: keras.Model, discriminator: keras.Model):
        """Creates a new C-VGAN model, that consists of two adversarial networks.

        :param generator: Generator network.
        :param discriminator: Discriminator network.
        """
        super(VGANConditional, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        """Calls the generator model on a set of inputs.

        :param inputs: Input for generator.
        :param training: Whether network should be run in training mode or inference mode.
        :param mask: Tensor mask; not implemented and therefore discarded.
        :return: Output of generator (generated videos).
        """
        return self.generator(inputs, training=training)

    # noinspection PyMethodOverriding
    def compile(self, g_optimizer: Optimizer, d_optimizer: Optimizer,
                g_loss_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
                d_loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]):
        """Configures the models for training.

        THIS METHOD HAS A DIFFERENT SIGNATURE THAN THE OVERRIDDEN METHOD! DO NOT USE THIS MODEL IN ITS ORIGINAL CONTEXT!

        :param g_optimizer: Generator optimizer instance.
        :param d_optimizer: Discriminator optimizer instance.
        :param g_loss_fn: Function that computes the generator loss.
        :param d_loss_fn: Function that computes the discriminator loss.
        """
        super(VGANConditional, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")

    def train_step(self, data) -> Mapping[str, float]:
        """Computes a single train step, i.e. training over one batch.

        :param data: Data tuple of a single batch.
            - input_videos - Beginning/Snippet of actual video clips. Input for generator.
            - real_videos  - Actual video clips. Input for discriminator.
        :return: Dictionary of metrics.
            - g_loss - Averaged generator loss for current epoch.
            - d_loss - Averaged discriminator loss for current epoch.
        """
        input_videos, real_videos = data

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_videos = self.generator(input_videos, training=True)

            real_output = self.discriminator(real_videos, training=True)
            fake_output = self.discriminator(generated_videos, training=True)

            g_loss = self.g_loss_fn(fake_output, input_videos, generated_videos)
            d_loss = self.d_loss_fn(real_output, fake_output)

        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {"g_loss": self.g_loss_tracker.result(), "d_loss": self.d_loss_tracker.result()}

    def test_step(self, data) -> Mapping[str, float]:
        """Computes a single evaluation step, i.e. evaluation over one batch.

        :param data: Data tuple of a single batch.
            - input_videos - Beginning/Snippet of actual video clips. Input for generator.
            - real_videos  - Actual video clips. Input for discriminator.
        :return: Dictionary of metrics.
            - g_loss - Averaged generator loss for current epoch.
            - d_loss - Averaged discriminator loss for current epoch.
        """
        input_videos, real_videos = data

        generated_videos = self.generator(input_videos, training=False)

        real_output = self.discriminator(real_videos, training=False)
        fake_output = self.discriminator(generated_videos, training=False)

        g_loss = self.g_loss_fn(fake_output, input_videos, generated_videos)
        d_loss = self.d_loss_fn(real_output, fake_output)

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {"g_loss": self.g_loss_tracker.result(), "d_loss": self.d_loss_tracker.result()}

    @property
    def metrics(self):
        """Returns the model's metrics added using `compile`. Metrics are reset every epoch."""
        return [self.g_loss_tracker, self.d_loss_tracker]


"""Collection of builder functions for pre-configured (sub-)models of the generative adversarial network for video 
(VGAN) by Vondrick et al. (http://www.cs.columbia.edu/~vondrick/tinyvideo/) and our adjustments to it. Merely the 
general concept of a two stream generator model and its one stream discriminator counterpart was kept from the original
paper. 

The code originally appeared in the different notebooks found in `src/models/` and in each function is a reference to 
the notebook of its first appearance. There you will also find a more detailed explanation regarding structure and the
parameters.
"""


def make_encoder_foreground_stream(inputs: tf.Tensor) -> tf.Tensor:
    """Creates the tensor that encodes the the input frames to the foreground (3D) latent space. This is done by
    applying several spatio-temporal convolutions until dimensions of 2x2x4x1024 are reached.

    Usage:
        src/models/vgan_conditional_3d.ipynb

    :param inputs: Input tensor (shape=None,7,64,128,3) for the background encoder.
    :return: Tensor for the background encoder stream.
    """
    x: tf.Tensor = layers.Conv3D(filters=64, kernel_size=3, strides=(1, 2, 2),
                                 padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    assert x.shape.as_list() == [None, 7, 32, 64, 64]
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 16, 32, 128]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, strides=(1, 2, 2),
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 8, 16, 256]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters=512, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 2, 4, 8, 512]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters=1024, kernel_size=3, strides=(1, 2, 2),
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 2, 2, 4, 1024]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def make_encoder_background_stream(inputs: tf.Tensor) -> tf.Tensor:
    """Creates the tensor that encodes the the input frames to the background (2D) latent space. This is done by
    applying several spatio-temporal convolutions, until the time dimension has a value of 1, before the singular frame
    is downsampled even more using spatial convolutions, until a dimension of 2x4x1024 is reached.

    Usage:
        src/models/vgan_conditional_3d.ipynb

    :param inputs: Input tensor (shape=None,7,64,128,3) for the background encoder.
    :return: Tensor for the background encoder stream.
    """
    x: tf.Tensor = layers.Conv3D(filters=64, kernel_size=3, strides=2,
                                 padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    assert x.shape.as_list() == [None, 4, 32, 64, 64]
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 2, 16, 32, 128]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 1, 8, 16, 256]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Reshape((8, 16, 256))(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 8, 512]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=1024, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 2, 4, 1024]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def make_conditional_foreground_stream(inputs: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Creates the modified tensor for the foreground stream of the VGAN architecture. Unlike the encoder-less
    foreground stream, the conditional one has one layer missing at its beginning, upsampling the input directly to
    a format of 2x4x8x512 instead of staying in latent space for one additional step. This results in the usage of more
    convolutional channels on each subsequent layer (except the last one). The rest of the stream is like the default
    one, generating a space-time cuboid of 8x64x128x3 (frames, height, width, rgb-channels).

    Usage:
        src/models/vgan_conditional_3d_2.ipynb

    :param inputs: Input tensor (shape=None,2,2,4,channels) for foreground stream.
    :return: (f,m) with f being the tensor for the foreground stream and m being the one for the mask.
    """
    x: tf.Tensor = layers.Conv3DTranspose(filters=512, kernel_size=3, strides=(1, 2, 2),
                                          padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    assert x.shape.as_list() == [None, 2, 4, 8, 512]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3DTranspose(filters=256, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 8, 16, 256]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3DTranspose(filters=128, kernel_size=3, strides=(1, 2, 2),
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 16, 32, 128]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3DTranspose(filters=64, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 8, 32, 64, 64]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return make_foreground_and_mask_output(x)


def make_conditional_background_stream(inputs: tf.Tensor) -> tf.Tensor:
    """Creates the modified tensor for the background stream of the VGAN architecture. Unlike the encoder-less
    background stream, the conditional one has one layer missing at its beginning, upsampling the input directly to
    a format of 4x8x512 instead of staying in latent space for one additional step. This results in the usage of more
    convolutional channels on each subsequent layer (except the last one). The rest of the stream is like the default
    one, generating a space-time cuboid of 8x64x128x3 (frames, height, width, rgb-channels).

    Usage:
        src/models/vgan_conditional_3d_2.ipynb

    :param inputs: Input tensor (shape=None,2,4,channels) for background stream.
    :return: Tensor for the background stream.
    """
    x: tf.Tensor = layers.Conv2DTranspose(filters=512, kernel_size=3, strides=2,
                                          padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    assert x.shape.as_list() == [None, 4, 8, 512]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 8, 16, 256]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 16, 32, 128]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 32, 64, 64]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return make_background_output(x)


def make_encoder_stream(inputs: tf.Tensor) -> tf.Tensor:
    """Creates the tensor that encodes the input frame to latent space for both foreground and background stream of the
    VGAN architecture. This consists of several strided 2D convolutions, downsampling the frame to latent space --
    2x4x1024. Only used when VGAN is used to creature future frames (C-VGAN).

    Usage:
        src/models/vgan_conditional.ipynb

    :param inputs: Input tensor (shape=None,64,128,3) for the encoder.
    :return: Tensor for the encoder stream.
    """
    x: tf.Tensor = layers.Conv2D(filters=64, kernel_size=3, strides=2,
                                 padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    assert x.shape.as_list() == [None, 32, 64, 64]
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 16, 32, 128]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=256, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 8, 16, 256]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=512, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 8, 512]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=1024, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 2, 4, 1024]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def make_generator_foreground_stream(inputs: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Creates the two tensors for the foreground stream of the VGAN architecture. This consist of multiple fractional
    strided 3D convolutions, upsampling the input to the desired video format -- 8x64x128x3 (frames, height, width,
    rgb-channels)

    Usage:
        src/models/vgan.ipynb

    :param inputs: Input tensor (shape=None,2,2,4,channels) for foreground stream.
    :return: (f,m) with f being the tensor for the foreground stream and m being the one for the mask.
    """
    x: tf.Tensor = layers.Conv3DTranspose(filters=512, kernel_size=3, strides=1,
                                          padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    assert x.shape.as_list() == [None, 2, 2, 4, 512]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3DTranspose(filters=256, kernel_size=3, strides=(1, 2, 2),
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 2, 4, 8, 256]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3DTranspose(filters=128, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 8, 16, 128]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3DTranspose(filters=64, kernel_size=3, strides=(1, 2, 2),
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 16, 32, 64]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3DTranspose(filters=32, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 8, 32, 64, 32]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return make_foreground_and_mask_output(x)


def make_generator_background_stream(inputs: tf.Tensor) -> tf.Tensor:
    """Creates the tensor for the background stream of the VGAN architecture. Unlike the foreground stream, only 2D
    fractional strided convolutions are used to upsample the background image, because the background is static (still
    camera is assumed), to the desired format, before the frame is replicated across the time dimension to get a space-
    time cuboid of 8x64x128x3 (frames, height, width, rgb-channels).

    Usage:
        src/models/vgan.ipynb

    :param inputs: Input tensor (shape=None,2,4,channels) for background stream.
    :return: Tensor for the background stream.
    """
    x: tf.Tensor = layers.Conv2DTranspose(filters=512, kernel_size=3, strides=1,
                                          padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    assert x.shape.as_list() == [None, 2, 4, 512]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 8, 256]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 8, 16, 128]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 16, 32, 64]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                               padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 32, 64, 32]
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return make_background_output(x)


def make_foreground_and_mask_output(inputs: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Helper function for the end of the foreground&mask stream; for each of the two, one final 3D fractional strided
    convolution is applied to achieve the final output shape. The mask tensor undergoes one final transformation, the
    replication of its singleton channel to match the RGB color channels of foreground and background.

    Usage:
        src/models/vgan.ipynb

    :param inputs: Input tensor (shape=None,8,32,64,channels) of foreground stream.
    :return: Tensor for the foreground and its mask.
    """
    # Foreground (f)
    f: tf.Tensor = layers.Conv3DTranspose(filters=3, kernel_size=3, strides=(1, 2, 2), activation="tanh",
                                          padding="same", use_bias=False, kernel_initializer="glorot_uniform")(inputs)
    assert f.shape.as_list() == [None, 8, 64, 128, 3]

    # Mask (m)
    m: tf.Tensor = layers.Conv3DTranspose(filters=1, kernel_size=3, strides=(1, 2, 2), activation="sigmoid",
                                          activity_regularizer=keras.regularizers.l1(l=0.1),
                                          padding="same", use_bias=False, kernel_initializer="glorot_uniform")(inputs)
    assert m.shape.as_list() == [None, 8, 64, 128, 1]

    # Replicate (m) across channel dimension
    m = tf.tile(m, [1, 1, 1, 1, 3])
    assert m.shape.as_list() == [None, 8, 64, 128, 3]

    return f, m


def make_background_output(inputs: tf.Tensor) -> tf.Tensor:
    """Helper function for the end of a background stream; after one final 2D fractional strided convolution the final
    shape of the background image is reached. Then the frame is replicated across the time dimension to get a space-
    time cuboid of 8x64x128x3 (frames, height, width, rgb-channels).

    Usage:
        src/models/vgan.ipynb

    :param inputs: Input tensor (shape=None,32,64,channels) of background stream.
    :return: Tensor for the background.
    """
    b = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation="tanh",
                               padding="same", use_bias=False, kernel_initializer="glorot_uniform")(inputs)
    assert b.shape.as_list() == [None, 64, 128, 3]

    # Replicate (b) across time dimension
    b = layers.Flatten()(b)
    assert b.shape.as_list() == [None, 64 * 128 * 3]
    b = layers.RepeatVector(n=8)(b)
    assert b.shape.as_list() == [None, 8, 64 * 128 * 3]
    b = layers.Reshape((8, 64, 128, 3))(b)
    assert b.shape.as_list() == [None, 8, 64, 128, 3]

    return b


def make_generator_stream_combiner(f: tf.Tensor, m: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Creates tensor for the result of the entire video generator. The three given arguments are merged by masking
    different parts of foreground and background, before adding these two together (G = m * f + ( 1 - m ) * b ). Format
    of the resulting space-time cuboid is the same as for the three parameters (8x64x128x3).

    Usage:
        src/models/vgan.ipynb

    :param f: Foreground tensor.
    :param m: Mask tensor.
    :param b: Background tensor.
    :return: Combined output tensor (i.e. the generated video)
    """
    m_mult_f = layers.Multiply()([m, f])
    inv_m = layers.Subtract()([tf.ones_like(m), m])
    inv_m_mult_b = layers.Multiply()([inv_m, b])
    outputs: tf.Tensor = layers.Add()([m_mult_f, inv_m_mult_b])
    assert outputs.shape.as_list() == [None, 8, 64, 128, 3]

    return outputs


def make_discriminator_model(filters=32, dropout_rate=0.) -> keras.Model:
    """Creates a preset discriminator model that distinguishes real videos from fake (generated) ones, for a resolution
    of 8x64x128x3.

    Usage:
        src/models/vgan.ipynb

    :param filters: Number of convolutional filters on the first layer. Will be used as baseline for all other conv
    layers. Serves as an empowerment/impairment parameter for the model.
    :param dropout_rate: Fraction of the units per convolutional layer to drop. Serves as an empowerment/impairment
    parameter for the model.
    :return: Discriminator model.
    """
    inputs: tf.Tensor = keras.Input(shape=(8, 64, 128, 3))

    x: tf.Tensor = layers.Conv3D(filters=filters, kernel_size=3, strides=(1, 2, 2),
                                 padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    assert x.shape.as_list() == [None, 8, 32, 64, filters]
    x = layers.LeakyReLU()(x)
    if dropout_rate != 0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv3D(filters=filters * 2, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 16, 32, filters * 2]
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    if dropout_rate != 0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv3D(filters=filters * 4, kernel_size=3, strides=(1, 2, 2),
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 4, 8, 16, filters * 4]
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    if dropout_rate != 0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv3D(filters=filters * 8, kernel_size=3, strides=2,
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 2, 4, 8, filters * 8]
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    if dropout_rate != 0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv3D(filters=filters * 16, kernel_size=3, strides=(1, 2, 2),
                      padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    assert x.shape.as_list() == [None, 2, 2, 4, filters * 16]
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    if dropout_rate != 0:
        x = layers.Dropout(dropout_rate)(x)

    # Get binary classification
    outputs: tf.Tensor = layers.Flatten()(x)
    assert outputs.shape.as_list() == [None, 2 * 2 * 4 * filters * 16]
    outputs = layers.Dense(1)(outputs)
    assert outputs.shape.as_list() == [None, 1]

    return keras.Model(inputs=inputs, outputs=outputs, name="vgan_discriminator")
