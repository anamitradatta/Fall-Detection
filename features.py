# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer
data. We recommend using helper functions like _compute_mean_features(window) to
extract individual features.
As a side note, the underscore at the beginning of a function is a Python
convention indicating that the function has private access (although in reality
it is still publicly accessible).
"""

import numpy as np
import math
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from time import sleep

def _mag(window):
    # compute the magnitude signal
    return [math.sqrt(x**2+y**2+z**2) for x, y, z in window]


def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    return np.mean(window, axis=0)


def _compute_median_features(window):
    """
    Compute median x, y, and z acceleration over given window.
    """
    return np.median(window, axis=0)


def _compute_std_features(window):
    """Compute the standard deviation of x, y, and z acceleration over window.
    """
    mag = [math.sqrt(x**2+y**2+z**2) for x, y, z in window]
    return np.std(mag, axis=0)

def _compute_max_magnitude_features(window):
    mag = [math.sqrt(x**2+y**2+z**2) for x, y, z in window]
    return np.amax(mag)


def _compute_entropy_features(window):

    mag = [math.sqrt(x**2+y**2+z**2) for x, y, z in window]

    hist, bin_edges = np.histogram(mag, bins=5, density=True)
   
    # calculate entropy here

    prob = np.diff(bin_edges)*hist
    entropy = 0
    for p in prob:
        if (p > 0):
            entropy += p*math.log2(p)

    return -entropy


def _compute_peak_features(window):

    mag = [math.sqrt(x**2+y**2+z**2) for x, y, z in window]

    ind, _ = find_peaks(mag, height=np.mean(mag)+1, prominence=1)
   
    return len(ind)


def _compute_peak_freq_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    
    mag = [math.sqrt(x**2+y**2+z**2) for x, y, z in window]
    
    sig = np.fft.rfft(mag, axis=0)
    rsig = sig.real.astype(float)
    ind, _ = find_peaks(rsig, prominence=1)
    
    if len(ind) == 0:
        return 0
    else:
        return rsig[ind[0]]


def _compute_n_peak_freq_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    
    mag = [math.sqrt(x**2+y**2+z**2) for x, y, z in window]
    
    sig = np.fft.rfft(mag, axis=0)
    rsig = sig.real.astype(float)
    
    ind, _ = find_peaks(rsig, prominence=1)
    
    return len(ind)


def extract_features(window):
    """
    Here is where you will extract your features from the data over
    the given window. We have given you an example of computing
    the mean and appending it to the feature vector.
    """

    x = []
    feature_names = []

    x.append(_compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")
    
    # convert the list of features to a single 1-dimensional vector
    feature_vector = np.concatenate(x, axis=0)
    
    #call functions to compute other features. Append the features to x and the names of these features to feature_names

    max_val = _compute_max_magnitude_features(window)
    feature_names.append("max_value")
    num_peaks = _compute_peak_features(window)
    feature_names.append("num_of_peaks")
    entropy = _compute_entropy_features(window)
    feature_names.append("entropy")
    
    num_peaks_fft = _compute_n_peak_freq_features(window)
    peak_fft = _compute_peak_freq_features(window)
    stdev = _compute_std_features(window)
    feature_names.append("num_of_peaks_fft")
    feature_names.append("peak_freq_fft")
    feature_names.append("stdev")
    
    feature_vector = np.append(feature_vector, max_val)
    feature_vector = np.append(feature_vector, num_peaks)
    feature_vector = np.append(feature_vector, entropy)
    feature_vector = np.append(feature_vector, num_peaks_fft)
    feature_vector = np.append(feature_vector, peak_fft)
    feature_vector = np.append(feature_vector, stdev)
    
    return feature_names, feature_vector