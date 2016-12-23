'''
processing.py
author: Luke de Oliveira, July 2015 

Simple utilities for processing the junk that comes out of the ntuple event generation.
'''

import numpy.linalg as la
import numpy as np
from .jettools import rotate_jet, flip_jet, plot_mean_jet


def angle_from_vec(v1, v2):
    # cosang = np.dot(v1, v2)
    cosang = v1
    sinang = la.norm(np.cross(np.array([1.0, 0.0]), np.array([v1, v2])))
    return np.arctan2(sinang, cosang)


def buffer_to_jet(entry, tag=0, side='r', max_entry=None, pix=25):
    """
    Takes an *single* entry from an structured ndarray, i.e., X[i], 
    and a tag = {0, 1} indicating if its a signal entry or not. 
    The parameter 'side' indicates which side of the final 
    jet image we want the highest energy.

    The `entry` must have the following fields (as produced by event-gen)
        * Intensity
        * PCEta, PCPhi
        * LeadingPt
        * LeadingEta
        * LeadingPhi
        * SubLeadingEta
        * SubLeadingPhi
        * LeadingM
        * DeltaR
        * Tau32
        * Tau21
        * Tau{n} for n = 1, 2, 3 
    """

    if (entry['SubLeadingEta'] < -10) | (entry['SubLeadingPhi'] < -10):
        e, p = (entry['PCEta'], entry['PCPhi'])
    else:
        e, p = (entry['SubLeadingEta'], entry['SubLeadingPhi'])

    angle = np.arctan(p / e) + 2.0 * np.arctan(1.0)

    if (-np.sin(angle) * e + np.cos(angle) * p) > 0:
        angle += -4.0 * np.arctan(1.0)

    # image = flip_jet(rotate_jet(np.array(entry['Intensity']), -angle, normalizer=4000.0, dim=pix), side)

    normalizer = 4000.0
    image = flip_jet(
        rotate_jet(np.array(entry['Intensity']), -angle,
                   normalizer=normalizer, dim=pix),
        side
    )
    # e_norm = np.linalg.norm(image)
    e_norm = 1.0 / normalizer
    return (
        (image / e_norm).astype('float32'),
        np.float32(tag),
        np.float32(entry['LeadingPt']),
        np.float32(entry['LeadingEta']),
        np.float32(entry['LeadingPhi']),
        np.float32(entry['LeadingM']),
        np.float32(entry['DeltaR']),
        np.float32(entry['Tau32']),
        np.float32(entry['Tau21']),
        np.float32(entry['Tau1']),
        np.float32(entry['Tau2']),
        np.float32(entry['Tau3'])
    )


def is_signal(f, matcher='wprime'):
    """
    Takes as input a filename and a string to match. If the 
    'matcher' string is found in the filename, the file is 
    taken to be a signal file.
    """
    key = matcher.lower().replace(' ', '').replace('-', '')
    if key in f.lower().replace(' ', '').replace('-', ''):
        return 1.0
    return 0.0
