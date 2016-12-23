'''
file: jettools.py
author: Luke de Oliveira, Mar. 2015 

Provides support for jet rotation and flipping

sukiyaki DL
'''
import numpy as np
import skimage.transform as sk
# import PIL
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def rotate_jet(jet, angle, in_radians=True, normalizer=None, dim=25):
    """
    Take an *flat* unrotated in the form of an array from Mediator.exe, and an angle, 
    and rotates the jet to that angle using a passively rotated cubic spline 
    interpolation
    """
    im = jet.reshape((dim, dim))
    if normalizer is not None:
        np.clip(im, -1, normalizer, out=im)
        im = np.flipud(im.T) / normalizer
    else:
        im = np.flipud(im.T)

    if in_radians:
        angle = np.rad2deg(angle)

    # return ndimage.interpolation.rotate(im, angle, reshape=False, order=3)
    # i = PIL.Image.fromarray(im).rotate(angle, resample=PIL.Image.BICUBIC)
    # return np.array(i)
    return sk.rotate(im, angle, order=3)


def flip_jet(jet, pool='r'):
    """
    Takes a rotated jet (25, 25) from rotate_jet() and flips it across the
    vertical axis according to whether you want the right side
    (pool = {r, R, right, Right, ...}) or the 
    left side (pool = {l, L, Left, left, ...}) to contain the most energy.
    """
    weight = jet.sum(axis=0)

    halfway = jet.shape[0] / 2.
    l, r = np.int(np.floor(halfway)), np.int(np.ceil(halfway))
    l_weight, r_weight = np.sum(weight[:l]), np.sum(weight[r:])

    if ('r' in pool.lower()) and ('l' in pool.lower()):
        raise ValueError('Jet pooling side must have l -OR- r in the name.')
    if 'r' in pool.lower():
        if r_weight > l_weight:
            return jet
        return np.fliplr(jet)
    elif 'l' in pool.lower():
        if l_weight > r_weight:
            return jet
        return np.fliplr(jet)
    else:
        raise ValueError('Jet pooling side must have l -OR- r in the name.')


def plot_mean_jet(rec, field='image', title='Average Jet Image'):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.mean(rec[field], axis=0),  norm=LogNorm(
        vmin=0.00001, vmax=1), interpolation='nearest')
    plt.title(r'' + title)
    return fig


def plot_jet(rec, title='Jet Image', log=True):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    if log:
        im = ax.imshow(rec,  norm=LogNorm(
            vmin=0.00001, vmax=1), interpolation='nearest')
    else:
        im = ax.imshow(rec, interpolation='nearest')
    plt.title(r'' + title)
    return fig
