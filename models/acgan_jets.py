"""
CPU (TF) -> 35 min / epoch
GeForce GT 750M (TH) -> 16 min / epoch
Titan X Maxwell Arch. (TF) ~ 1 min / epoch
Titan X Maxwell Arch. (TH) ~ 1.5 min / epoch
"""

from PIL import Image

from keras.backend import set_image_dim_ordering
from keras.datasets import mnist
from keras.initializations import normal as _normal_init
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.utils.generic_utils import Progbar
import numpy as np

np.random.seed(1337)

set_image_dim_ordering('th')


def adam_config():
    # parameters suggested in [PAPER LINK]
    return Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)


def build_generator(latent_size):
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size))
    cnn.add(LeakyReLU())
    cnn.add(Dense(128 * 7 * 7))
    cnn.add(LeakyReLU())
    cnn.add(Reshape((128, 7, 7)))

    # upsample to (..., 64, 14, 14)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(64, 5, 5, border_mode='same', init='glorot_normal'))
    cnn.add(LeakyReLU())

    # upsample to (..., 64, 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))

    # valid conv to (..., 32, 25, 25)
    cnn.add(Convolution2D(32, 4, 4, border_mode='valid', init='glorot_normal'))
    cnn.add(LeakyReLU())

    # take a channel axis reduction to (..., 1, 25, 25)
    cnn.add(Convolution2D(1, 2, 2, border_mode='same', init='glorot_normal'))
    cnn.add(Activation('relu'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1, 1), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(2, latent_size, input_length=1,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(input=[latent, image_class], output=fake_image)


def build_discriminator():
    # build a relatively standard conv net
    cnn = Sequential()

    cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                          input_shape=(1, 25, 25)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 25, 25))

    features = cnn(image)

    # fake output tracks binary fake / not-fake, and the auxiliary requires
    # reconstruction of latent features, in this case, labels
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(1, activation='sigmoid', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])

if __name__ == '__main__':
    nb_epochs = 30
    batch_size = 100
    latent_size = 100
    nb_labels = 2

    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(adam_config(), ['binary_crossentropy',
                                          'binary_crossentropy'])

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(adam_config(), 'binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1, 1), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(input=[latent, image_class], output=[fake, aux])

    combined.compile(adam_config(), ['binary_crossentropy',
                                     'binary_crossentropy'])

    discriminator.trainable = True

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]

    d = np.load('/home/lukedeo/scratch/data/gan/jetimages.npy')
    X = d['image']
    y = d['signal']

    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    for epoch in range(nb_epochs):
        print "Epoch {} of {}".format(epoch + 1, nb_epochs)

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        # for index in tqdm(range(nb_batches), unit=' batches'):
        for index in xrange(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, nb_labels, batch_size)

            # ...and a batch of fake images, using the generated labels as a
            # conditioner
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1, 1))], verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # make new noise
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, nb_labels, 2 * batch_size)

            discriminator.trainable = False

            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1, 1))], [trick, sampled_labels]))

            discriminator.trainable = True

        print "\nTesting for epoch {}:".format(epoch + 1)

        # evaluate the testing loss

        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (nb_test, latent_size))

        # get a batch of real images
        image_batch = X_train
        label_batch = y_train

        # sample some labels from p_c
        sampled_labels = np.random.randint(0, nb_labels, nb_test)

        # ...and a batch of fake images, using the generated labels as a
        # conditioner
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1, 1))], verbose=False)

        X = np.concatenate((image_batch, generated_images))
        y = np.array([1] * nb_train + [0] * nb_test)
        aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)

        # make new noise
        noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, nb_labels, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1, 1))], [trick, sampled_labels], verbose=False)

        report = {
            'generator': {
                'train': np.mean(np.array(epoch_gen_loss), axis=0),
                'test': generator_test_loss
            },
            'discriminator': {
                'train': np.mean(np.array(epoch_disc_loss), axis=0),
                'test': discriminator_test_loss,
            }
        }

        HEADING = '{0:<22s} | {1:4s} | {2:15s} | {3:5s}'
        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'

        print HEADING.format('component', *discriminator.metrics_names)
        print '-' * 65
        for m in ['generator', 'discriminator']:
            for p in ['train', 'test']:
                print ROW_FMT.format('{m} ({p})'.format(m=m, p=p), *report[m][p])

        generator.save_weights(
            'models/generator_epoch_{}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'models/discriminator_epoch_{}.hdf5'.format(epoch), True)

        # noise = np.random.uniform(-1, 1, (nb_labels0, latent_size))

        # sampled_labels = np.array([
        #     [i] * 50 for i in xrange(2)
        # ]).reshape(-1, 1, 1)

        # # get a batch to display
        # generated_images = generator.predict(
        #     [noise, sampled_labels], verbose=0)

        # img = (np.concatenate([r.reshape(-1, 25)
        #                        for r in np.split(generated_images, 10)
        #                        ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        # Image.fromarray(img).save(
        #     "plots/epoch_{}_generated.png".format(epoch))
