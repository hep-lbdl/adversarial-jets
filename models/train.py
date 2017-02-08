#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: train.py
description: training script for [arXiv/1701.05927]
author: Luke de Oliveira (lukedeoliveira@lbl.gov)
"""

from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import argparse

from six.moves import range
import sys

from h5py import File as HDF5File
import numpy as np


def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run LAGAN training from [arXiv/1701.05927]. '
        'Sensible defaults come from [arXiv/1511.06434]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nb-epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch-size', action='store', type=int, default=100,
                        help='batch size per update')
    parser.add_argument('--latent-size', action='store', type=int, default=200,
                        help='size of random N(0, 1) latent space to sample')

    # Adam parameters suggested in [arXiv/1511.06434]
    parser.add_argument('--adam-lr', action='store', type=float, default=0.0002,
                        help='Adam learning rate')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    parser.add_argument('--dataset', action='store', type=str,
                        help='HDF5 or Numpy array to train from. If not '
                        'specified, will download directly from '
                        '[10.5281/zenodo.268592] into a Keras cache')

    parser.add_argument('--nb-points', action='store', type=int, default=90000,
                        help='Number points to use from the downloaded file')

    parser.add_argument('--prog-bar', action='store_true',
                        help='Whether or not to use a progress bar')

    parser.add_argument('--d-pfx', action='store',
                        defualt='params_discriminator_epoch_'
                        help='Default prefix for discriminator network weights')

    parser.add_argument('--g-pfx', action='store',
                        defualt='params_generator_epoch_'
                        help='Default prefix for generator network weights')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    results = parser.parse_args()

    # delay the imports so running train.py -h doesn't take 50 years
    import keras.backend as K

    K.set_image_dim_ordering('tf')

    from keras.layers import Input
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.utils.generic_utils import Progbar
    from sklearn.cross_validation import train_test_split

    from lagan.generator import generator as build_generator
    from lagan.discriminator import discriminator as build_discriminator

    # batch, latent size, and whether or not to be verbose with a progress bar
    nb_epochs = results.nb_epochs
    batch_size = results.batch_size
    latent_size = results.latent_size
    verbose = results.prog_bar

    nb_classes = 2

    adam_lr = results.adam_lr
    adam_beta_1 = results.adam_beta

    # build the discriminator
    print('Building discriminator')
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'binary_crossentropy']
    )

    # build the generator
    print('Building generator')
    generator = build_generator(latent_size)
    generator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )

    image_class = Input(shape=(1, ), name='combined_aux', dtype='int32')
    latent = Input(shape=(latent_size, ), name='combined_z')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(
        input=[latent, image_class],
        output=[fake, aux],
        name='combined_model'
    )

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'binary_crossentropy']
    )

    datafile = results.dataset

    # if we don't have the dataset, go fetch it from Zenodo, or re-find in the
    # Keras cache
    print('Loading data')
    if not datafile or os.path.isfile(datafile):

        from keras.utils.data_utils import get_file

        print('File not found or not specified. Downloading from Zenodo. '
              '(Or, falling back to cache if present)')

        # Info for downloading the dataset from Zenodo
        MD5_HASH = 'f9b11c46b6a0ff928bec2eccf865ecf0'
        DATAFILE = 'jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5'
        URL_TEMPLATE = 'https://zenodo.org/record/{record}/files/{filename}'

        print('MD5 verification: {}'.format(MD5_HASH))

        datafile = get_file(
            fname='lagan-jet-images.hdf5',
            origin=URL_TEMPLATE.format(record=269622, filename=DATAFILE),
            md5_hash=MD5_HASH
        )

    # You can pass in either HDF5 files or Numpy binary files - we default to
    # HDF5, but can fallback to numpy
    try:
        d = HDF5File(datafile, 'r')

        X, y = d['image'][:], d['signal'][:]

        ix = list(range(X.shape[0]))
        np.random.shuffle(ix)
        ix = ix[:results.nb_points]

        X, y = X[ix], y[ix]

    except IOError:
        print('Failure to read as HDF5, falling back to numpy')

        d = np.load(datafile, mmap_mode='r')
        ix = list(range(d.shape[0]))
        np.random.shuffle(ix)
        ix = ix[:results.nb_points]
        d = np.array(d[ix])

        X, y = d['image'], d['signal']

    # remove unphysical values
    X[X < 1e-3] = 0

    # we don't really need validation data as it's a bit meaningless for GANs,
    # but since we have an auxiliary task, it can be helpful to debug mode
    # collapse to a particularly signal or background-like image
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

    # tensorflow ordering
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    # scale the pT levels by 100 (help neural nets w/ dynamic range - they
    # need all the help they can get)
    X_train = X_train.astype(np.float32) / 100
    X_test = X_test.astype(np.float32) / 100

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    print('processed {}/{} batches'.format(index + 1, nb_batches))

            # generate a new batch of noise
            noise = np.random.normal(0, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c (note: we have a flat prior here, so
            # we can just sample randomly)
            sampled_labels = np.random.randint(0, nb_classes, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            # see if the discriminator can figure itself out...
            real_batch_loss = discriminator.train_on_batch(
                image_batch, [bit_flip(np.ones(batch_size)), label_batch]
            )

            # note that a given batch should have either *only* real or *only* fake,
            # as we have both minibatch discrimination and batch normalization, both
            # of which rely on batch level stats
            fake_batch_loss = discriminator.train_on_batch(
                generated_images,
                [bit_flip(np.zeros(batch_size)), bit_flip(sampled_labels)]
            )

            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, real} labels to say
            # real
            trick = np.ones(batch_size)

            gen_losses = []

            # we do this twice simply to match the number of batches per epoch used to
            # train the discriminator
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_labels = np.random.randint(0, nb_classes, batch_size)

                gen_losses.append(combined.train_on_batch(
                    [noise, sampled_labels.reshape((-1, 1))],
                    [trick, bit_flip(sampled_labels, 0.09)]
                ))

            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # generate a new batch of noise
        noise = np.random.normal(0, 1, (nb_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, nb_classes, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False, batch_size=batch_size)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.normal(0, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, nb_classes, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False, batch_size=batch_size)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance. **NOTE** that these values
        # don't mean a whole lot, but they can be helpful for diagnosing *serious*
        # instabilities with the training
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights('{0}{1:03d}.hdf5'.format(results.g_pfx, epoch),
                               overwrite=True)
        discriminator.save_weights('{0}{1:03d}.hdf5'.format(results.d_pfx, epoch),
                                   overwrite=True)

    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))
