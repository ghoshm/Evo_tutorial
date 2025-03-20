import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import copy

plt.style.use("./style_sheet.mplstyle")


class RecurrentNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        """
        Creates an RNN model.
        Arguments:
            n_inputs: number of input units.
            n_hidden: number of hidden units.
            n_outputs: number of output units.
        """
        # Units
        self.units = np.arange(n_inputs + n_hidden + n_outputs)
        self.unit_types = np.concatenate(
            [np.zeros(n_inputs), np.ones(n_hidden), np.ones(n_outputs) * 2]
        )  # Input, hidden and output units are labelled as 0, 1, 2

        # Connections
        self.connections = np.empty(
            (0, 3)
        )  # formatted as output unit, input unit, weight

    def forward(self, inputs):
        """
        Forward through the RNN model.
        Arguments:
            inputs: a numpy array (inputs x time).
        """
        assert sum(self.unit_types == 0) == inputs.shape[0], "Mismatched inputs"

        # Reset network
        self.activations_t_1 = np.zeros_like(self.units)
        self.activations_t = np.zeros_like(self.units)

        # Run
        for time in range(inputs.shape[1]):

            for unit in self.units:
                if self.unit_types[unit] == 0:
                    self.activations_t[unit] = inputs[unit, time]
                else:
                    self.activations_t[unit] = 0

                if sum(self.connections[:, 1] == unit) > 0:
                    for o, i, w in self.connections[self.connections[:, 1] == unit]:
                        self.activations_t[unit] += self.activations_t_1[int(o)] * w

            self.activations_t_1 = np.copy(self.activations_t)

        return self.activations_t[self.unit_types == 2] + np.random.normal(
            loc=1e-9, scale=1, size=sum(self.unit_types == 2)
        )

    def add_connections(self, number):
        """
        Add (number) connections to the RNN.
            Note that only a single connection is permitted per unit pair.
        """

        for _ in range(number):
            o = np.random.choice(self.units)
            i = np.random.choice(self.units)
            w = np.random.uniform(-1, 1)
            self.connections = np.vstack((self.connections, np.array([o, i, w])))
            _, idx = np.unique(self.connections[:, :2], axis=0, return_index=True)
            self.connections = self.connections[idx]

    def add_nodes(self, number):
        """
        Add (number) nodes to the RNN.
        """

        for _ in range(number):
            self.units = np.append(self.units, self.units[-1] + 1)
            self.unit_types = np.append(self.unit_types, 1)


# def plot_approximation(network, neuron_spike_times, neuron_spike_idx, input_I):
#     """"""
#     plt.subplots(figsize=(15, 5))

#     for idx_repeat in tqdm(range(10)):
#         neuron_spikes = neuron_spike_times[
#             neuron_spike_idx == idx_repeat
#         ]  # get "real" spikes
#         model_spikes = network.run_and_readout_spikes(
#             input_I[idx_repeat].reshape(1, -1)
#         )

#         plt.scatter(
#             neuron_spikes,
#             idx_repeat * np.ones(len(neuron_spikes)) + 0.2,
#             marker=".",
#             color="xkcd:purple",
#             label="HH model (spikes)" if idx_repeat == 0 else None,
#         )
#         plt.scatter(
#             model_spikes,
#             idx_repeat * np.ones(len(model_spikes)),
#             marker=".",
#             color="xkcd:dark seafoam green",
#             label="Network (spikes)" if idx_repeat == 0 else None,
#         )

#         plt.xlabel("Time")
#         plt.yticks([])
#         plt.ylabel("Repeat")
#         plt.legend()


# def van_rossum_distance(t0, t1, duration, tau_vr=5, dt=0.1):
#     """"""
#     n = int(np.round(duration / dt))
#     x0 = np.zeros(n)
#     x1 = np.zeros(n)
#     for x, t in [(x0, t0), (x1, t1)]:
#         x[np.array(np.round(t / dt), dtype=int)] = 1
#     nk = int(np.round(3 * tau_vr / dt))
#     if 2 * nk + 1 > n:
#         nk = (n - 1) // 2
#     T = np.arange(-nk, nk + 1) * dt
#     kernel = np.exp(-T / tau_vr) / tau_vr
#     for x in [x0, x1]:
#         x[:] = np.convolve(x, kernel, "same")

#     return np.sqrt(np.sum((x0 - x1) ** 2 * dt) / tau_vr)

# def mean_vr_distance(
#     network,
#     neuron_spike_times,
#     neuron_spike_idx,
#     input_I,
#     dt=0.1,
#     disable_reporting=True,
# ):
#     """"""
#     duration = input_I.shape[1] * dt

#     # Convert (t,i) format to list
#     neuron_spikes, model_spikes = [], []
#     for idx_repeat in tqdm(range(input_I.shape[0]), disable=disable_reporting):

#         # "Real" neuron
#         neuron_spikes.append(neuron_spike_times[neuron_spike_idx == idx_repeat])

#         # Model
#         m_spikes = network.run_and_readout_spikes(input_I[idx_repeat].reshape(1, -1))
#         model_spikes.append(m_spikes)

#     # Distance
#     d = 0
#     for t0, t1 in zip(neuron_spikes, model_spikes):
#         d += van_rossum_distance(t0, t1, duration, dt=dt)
#     d /= len(neuron_spikes)

#     if disable_reporting is False:
#         print("Mean vr distance: " + str(np.round(d, 2)))

#     return d
