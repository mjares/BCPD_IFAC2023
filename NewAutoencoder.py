from keras.models import load_model
import numpy as np
from scipy.stats import mode
from copy import copy


class AEThresholdClassifier:
    def __init__(self, model_path):
        self.scales = [0, 0]
        self.autoencoder = None  # Keras Model
        self.model_path = model_path
        self.Q = 0  # Reconstruction error threshold
        self.conf_interval = 0.0

    def load_autoencoder(self):
        self.autoencoder = load_model(self.model_path)

    def reconstruct(self, x, normalize=True):
        if normalize:
            x = self.scale(x)
        # x_ = self.autoencoder.predict(x.reshape([1, len(x)]), verbose=False)
        x_ = self.autoencoder.predict(x, verbose=False)
        return x_, x

    def predict(self, x, normalize=True):

        # Only for single samples
        x_, scaled_x = self.reconstruct(x, normalize)
        # error = np.mean(np.square(scaled_x - x_))
        error = np.mean(np.square(scaled_x - x_), axis=1)

        # if error - self.Q > 0:
        #     return 1
        # else:
        #     return 0
        return error - self.Q > 0, error

    def scale(self, x):
        return (x - self.scales[0]) / self.scales[1]


def time_horizon_analysis(modes, th):
    th_modes = copy(modes)
    for ii in range(th + 1, len(modes)):
        th_modes[ii] = mode(modes[ii - th:ii])[0]

    return th_modes


def CPD(modes):
    cpd = np.zeros_like(modes)
    for ii in range(1, len(modes)):
        if modes[ii] != modes[ii - 1]:
            cpd[ii] = 1
            # break
    return cpd, np.argwhere(cpd == 1)
