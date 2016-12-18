import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar


def basic_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                          input_shape=(1, 28, 28)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 28, 28))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])


def two_channel_discriminator():

    dnn = Sequential()
    dnn.add(Flatten(input_shape=(1, 25, 25)))

    # dnn.add(Dense(512, init='he_uniform'))
    # dnn.add(LeakyReLU())
    # dnn.add(Dropout(0.3))

    dnn.add(Dense(512, init='he_uniform'))
    dnn.add(LeakyReLU())
    dnn.add(Dropout(0.3))
    dnn.add(BatchNormalization(mode=2, axis=1))

    # dnn.add(Dense(512, init='he_uniform'))
    # dnn.add(LeakyReLU())
    # dnn.add(Dropout(0.3))

    dnn.add(Dense(256, init='he_uniform'))
    dnn.add(LeakyReLU())
    dnn.add(Dropout(0.3))

    dnn.add(Dense(256, init='he_uniform'))
    dnn.add(LeakyReLU())
    dnn.add(Dropout(0.2))

    cnn = Sequential()
    cnn.add(Convolution2D(32, 7, 7, border_mode='same', subsample=(2, 2),
                          input_shape=(1, 25, 25)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    cnn.add(BatchNormalization(mode=2, axis=1))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    cnn.add(BatchNormalization(mode=2, axis=1))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 25, 25))

    features = merge([dnn(image), cnn(image)], mode='concat', concat_axis=-1)

    # fake output tracks binary fake / not-fake, and the auxiliary requires
    # reconstruction of latent features, in this case, labels
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(1, activation='sigmoid', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])


def two_channel_seperate_discriminator(return_aux=False):

    aux_net = Sequential()
    aux_net.add(Flatten(input_shape=(1, 25, 25)))

    # aux_net.add(Dense(512, init='he_uniform'))
    # aux_net.add(LeakyReLU())
    # aux_net.add(Dropout(0.3))

    aux_net.add(Dense(512, init='he_uniform'))
    aux_net.add(LeakyReLU())
    aux_net.add(Dropout(0.3))

    # aux_net.add(Dense(512, init='he_uniform'))
    # aux_net.add(LeakyReLU())
    # aux_net.add(Dropout(0.3))

    aux_net.add(Dense(256, init='he_uniform'))
    aux_net.add(LeakyReLU())
    aux_net.add(Dropout(0.3))

    aux_net.add(Dense(256, init='he_uniform'))
    aux_net.add(LeakyReLU())
    aux_net.add(Dropout(0.2))

    aux_net.add(Dense(256, init='he_uniform'))
    aux_net.add(LeakyReLU())
    aux_net.add(Dropout(0.2))

    dnn = Sequential()
    dnn.add(Flatten(input_shape=(1, 25, 25)))

    # dnn.add(Dense(512, init='he_uniform'))
    # dnn.add(LeakyReLU())
    # dnn.add(Dropout(0.3))

    dnn.add(Dense(512, init='he_uniform'))
    dnn.add(LeakyReLU())
    dnn.add(Dropout(0.3))

    # dnn.add(Dense(512, init='he_uniform'))
    # dnn.add(LeakyReLU())
    # dnn.add(Dropout(0.3))

    dnn.add(Dense(256, init='he_uniform'))
    dnn.add(LeakyReLU())
    dnn.add(Dropout(0.3))

    dnn.add(Dense(256, init='he_uniform'))
    dnn.add(LeakyReLU())
    dnn.add(Dropout(0.2))

    cnn = Sequential()
    cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                          input_shape=(1, 25, 25)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 25, 25))

    features = merge([dnn(image), cnn(image)], mode='concat', concat_axis=-1)

    # fake output tracks binary fake / not-fake, and the auxiliary requires
    # reconstruction of latent features, in this case, labels
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(1, activation='sigmoid', name='auxiliary')(aux_net(image))

    if not return_aux:
        return Model(input=image, output=[fake, aux])
    return Model(input=image, output=[fake, aux]), Model(input=image, output=aux)
