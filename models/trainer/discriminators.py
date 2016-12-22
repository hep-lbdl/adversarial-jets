import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar


from tensor import minibatch_discriminator, minibatch_output_shape, DenseTensor


K.set_image_dim_ordering('tf')


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


def two_channel_discriminator(batch_size=100):

    image = Input(shape=(25, 25, 1))

    x = Flatten()(image)
    h1 = Dense(1024, init='he_uniform')(x)
    h1 = LeakyReLU()(h1)
    h1 = Dropout(0.3)(h1)

    h2 = Dense(1024, init='he_uniform')(h1)
    h2 = LeakyReLU()(h2)
    h2 = Dropout(0.3)(h2)

    h3 = Dense(1024, init='he_uniform')(h2)
    h3 = LeakyReLU()(h3)
    h3 = Dropout(0.3)(h3)

    h4 = Dense(1024, init='he_uniform')(h3)
    h4 = LeakyReLU()(h4)
    h4 = Dropout(0.3)(h4)

    h5 = Dense(1024, init='he_uniform')(h4)
    h5 = LeakyReLU()(h5)
    h5 = merge([Dropout(0.3)(h5), h1], mode='sum')

    dnn_out = LeakyReLU()(Dense(512, init='he_uniform')(h5))

    dnn = Model(input=image, output=dnn_out)

    # dnn = Sequential()
    # dnn.add(Flatten(input_shape=(25, 25, 1)))

    # # dnn.add(Dense(512, init='he_uniform'))
    # # dnn.add(LeakyReLU())
    # # dnn.add(Dropout(0.3))

    # dnn.add(Dense(1024, init='he_uniform'))
    # dnn.add(LeakyReLU())
    # dnn.add(Dropout(0.3))
    # # dnn.add(BatchNormalization(mode=2, axis=1))

    # # dnn.add(Dense(512, init='he_uniform'))
    # # dnn.add(LeakyReLU())
    # # dnn.add(Dropout(0.3))

    # dnn.add(Dense(1024, init='he_uniform'))
    # dnn.add(LeakyReLU())
    # dnn.add(Dropout(0.3))

    # dnn.add(Dense(512, init='he_uniform'))
    # dnn.add(LeakyReLU())
    # dnn.add(Dropout(0.3))

    cnn = Sequential()
    cnn.add(Convolution2D(32, 3, 3, border_mode='same',
                          init='he_uniform', input_shape=(25, 25, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same',
                          init='he_uniform', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    # cnn.add(BatchNormalization(mode=2, axis=-1))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same',
                          init='he_uniform', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    # cnn.add(BatchNormalization(mode=2, axis=-1))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same',
                          init='he_uniform', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    cnn.add(Dense(512, init='he_uniform'))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    image = Input(shape=(25, 25, 1))

    dnn_out = dnn(image)
    cnn_out = cnn(image)

    features = merge([dnn_out, cnn(image)], mode='concat', concat_axis=-1)

    # nb of features to obtain
    nb_features = 100

    # dim of kernel space
    vspace_dim = 10

    dnn_cmp_space = DenseTensor(nb_features, vspace_dim)(dnn_out)

    # concat the minibatch features with the normal ones
    features = merge([
        Lambda(minibatch_discriminator,
               output_shape=minibatch_output_shape)(dnn_cmp_space),
        features
    ], mode='concat')

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
