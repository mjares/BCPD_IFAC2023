import numpy as np


class CPDDataset:
    def __init__(self):
        self.episodes = []
        self.f_mag = []
        self.f_type = []
        self.changepoint = []
        self.no_episodes = 0

    def add_episode(self, traj_episode):
        self.no_episodes = self.no_episodes + 1
        new_entry = CPDDataset_Entry()
        new_entry.from_trajectory_dataset(traj_episode)
        self.episodes.append(new_entry)
        self.f_mag.append(new_entry.f_mag)
        self.f_type.append(new_entry.f_type)
        self.changepoint.append(new_entry.changepoint)


class CPDDataset_Entry:
    def __init__(self):
        self.features = None
        self.no_samples = 0
        self.stable_at_goal = False
        self.changepoint = 0
        self.f_mag = 0.0
        self.f_type = ''

    def from_trajectory_dataset(self, trajectory_dataset):
        self.no_samples = trajectory_dataset.weight.shape[0]
        self.f_mag = trajectory_dataset.conditions['Mag']
        self.f_type = trajectory_dataset.conditions['Fault']
        self.stable_at_goal = trajectory_dataset.stable_at_goal
        self.changepoint = trajectory_dataset.conditions['Times'][0]

        obs = trajectory_dataset.observations
        state = obs['state']
        control = obs['motor_command']
        # For trajectory_dataset: state ii is the result of applying control command ii to state ii - 1.
        control = np.delete(control, 0,  axis=0)
        next_state = state
        next_state = np.delete(next_state, 0,  axis=0)
        state = np.delete(state, -1,  axis=0)
        self.no_samples = self.no_samples - 1
        self.features = np.hstack([state, next_state, control])

    def trim(self, start=100, end=None):
        if end is not None and end < self.no_samples:
            self.no_samples = end
            self.features = self.features[:end, :]
        self.no_samples = self.no_samples - start
        self.features = np.delete(self.features, np.arange(start), axis=0)
        self.changepoint = self.changepoint - start


class CPDResults:
    def __init__(self):
        self.episode_results = []
        self.time_horizon = 0
        self.quantile = 0
        self.conf_interval = 0
        self.autoencoder_name = ''


class CPDEpisodeResults:
    def __init__(self):
        self.fault_mag = 0
        self.fault_type = ''
        self.starttime = 0
        self.no_samples = 0
        self.reconst_error = np.empty(1)
        self.mode_log = np.empty(1)
        self.mode_log_th = np.empty(1)
        self.changepoints = []
        self.changepoints_tstamps = []
