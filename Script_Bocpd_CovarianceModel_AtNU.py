import os
import pickle
from timeit import timeit

import numpy as np
import matplotlib.pyplot as plt

# BOCPD libs
from bayesian_changepoint_detection.priors import const_prior, geom_prior
from bayesian_changepoint_detection.offline_likelihoods import FullCovarianceLikelihood
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
from functools import partial

# Local libs
from NewAutoencoder import AEThresholdClassifier, time_horizon_analysis, CPD
from CPDResults import CPDResults, CPDEpisodeResults

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Params
show = True
basesavefilename = 'Results/CPDResults_AtNU_EpCount_200'

# Parameters
truncate = 1000
# Dataset
dFilename = 'CPDDataset_AtNU_EpCount_200.ds'
dsetFile = open(dFilename, 'rb')
dataset = pickle.load(dsetFile)
no_episodes = dataset.no_episodes

# Scaling
ae_model = f'Lat15_HL1_Leak10_E1000_Batch1K_95'
aeFile = open(ae_model + '.ae', 'rb')
ae_classifier = pickle.load(aeFile)

# Results
savefilename = f'{basesavefilename}_Bocpd_Covmodel_Trunc{truncate}_AtNU_EpCount_200.rs'
configResults = CPDResults()
for ee in range(no_episodes):
    print(f'Episode: {ee + 1}')
    feature_log = ae_classifier.scale(dataset.episodes[ee].features)
    starttime = dataset.episodes[ee].changepoint
    stepcount = dataset.episodes[ee].no_samples
    f_type = dataset.episodes[ee].f_type
    f_mag = dataset.episodes[ee].f_mag

    # CPD
    prior = partial(const_prior, p=1 / (stepcount + 1))
    Q_ifm, P_ifm, Pcp_ifm = offline_changepoint_detection(
        feature_log, prior, FullCovarianceLikelihood(),
        truncate=-truncate, verbose=False)

    # Plot
    fcm = np.exp(Pcp_ifm).sum(0)  # full covariance model
    # plt.figure()
    # plt.plot(fcm)
    # if f_type != 'Nominal':
    #     plt.plot([starttime, starttime], [np.min(fcm), np.max(fcm)])
    # plt.title(f'Fault: {f_type}: {f_mag}')

    # Episode Results
    epResults = CPDEpisodeResults()
    epResults.fault_mag = f_mag
    epResults.fault_type = f_type
    epResults.starttime = starttime
    epResults.no_samples = stepcount
    epResults.reconst_error = {'fcm': fcm, 'Q_ifm': Q_ifm, 'P_ifm': P_ifm, 'Pcp_ifm': Pcp_ifm}
    epResults.truncation = truncate
    epResults.prior_const = 1 / (stepcount + 1)
    configResults.episode_results.append(epResults)

Results = configResults
saveFile = open(savefilename, 'wb')
pickle.dump(Results, saveFile)
saveFile.close()
