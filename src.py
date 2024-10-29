import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class RecurrentNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        """"""
        # Units
        self.units = np.arange(n_inputs + n_hidden + n_outputs)
        self.unit_types = np.concatenate(
            [np.zeros(n_inputs), np.ones(n_hidden), np.ones(n_outputs) * 2]
        )

        # Activations
        self.unit_activations = [np.zeros(len(self.units)), np.zeros(len(self.units))]
        self.state = 0

        # Connections
        self.connections = np.empty((0, 3))

    def forward(self, inputs):
        """"""
        assert sum(self.unit_types == 0) == inputs.shape[0], "Mismatched inputs"

        # Reset network
        self.unit_activations = [self.unit_activations[s] * 0.0 for s in [0, 1]]
        outputs = []

        # Run
        for time in range(inputs.shape[1]):
            i_values = self.unit_activations[self.state]
            o_values = self.unit_activations[1 - self.state]
            self.state = 1 - self.state

            for unit in self.units:

                if self.unit_types[unit] == 0:
                    unit_input = inputs[unit, time]
                else:
                    unit_input = 0.0

                if len(self.connections) > 0:
                    for o, i, w in self.connections[self.connections[:, 1] == unit]:
                        unit_input += i_values[int(o)] * w

                o_values[unit] = 1 / (1 + np.exp(-unit_input))

            outputs.append(o_values[self.unit_types == 2])

        return np.array(outputs)

    def run_and_readout_spikes(self, inputs):
        """"""

        outputs = self.forward(inputs)
        outputs += np.random.rand(*outputs.shape) / 1000
        spikes = np.argwhere(outputs[:, 1] > outputs[:, 0]) * 0.1

        return np.array(spikes).reshape(-1)

    def add_connections(self, number):
        """"""

        for _ in range(number):
            o = np.random.choice(self.units)
            i = np.random.choice(self.units)
            w = np.random.uniform(-1, 1)
            self.connections = np.vstack((self.connections, np.array([o, i, w])))
            _, idx = np.unique(self.connections[:, :2], axis=0, return_index=True)
            self.connections = self.connections[idx]

    def add_nodes(self, number):
        """"""

        for _ in range(number):
            self.units = np.append(self.units, self.units[-1] + 1)
            self.unit_types = np.append(self.unit_types, 1)
            self.unit_activations = [
                np.zeros(len(self.units)),
                np.zeros(len(self.units)),
            ]
            self.state = 0


def plot_approximation(network, neuron_spike_times, neuron_spike_idx, input_I):
    """"""
    plt.subplots(figsize=(15, 5))

    for idx_repeat in tqdm(range(10)):
        neuron_spikes = neuron_spike_times[
            neuron_spike_idx == idx_repeat
        ]  # get "real" spikes
        model_spikes = network.run_and_readout_spikes(
            input_I[idx_repeat].reshape(1, -1)
        )

        plt.scatter(
            neuron_spikes,
            idx_repeat * np.ones(len(neuron_spikes)) + 0.2,
            marker=".",
            color="xkcd:purple",
            label="HH model (spikes)" if idx_repeat == 0 else None,
        )
        plt.scatter(
            model_spikes,
            idx_repeat * np.ones(len(model_spikes)),
            marker=".",
            color="xkcd:dark seafoam green",
            label="Network (spikes)" if idx_repeat == 0 else None,
        )

        plt.xlabel("Time")
        plt.yticks([])
        plt.ylabel("Repeat")
        plt.legend()


def van_rossum_distance(t0, t1, duration, tau_vr=5, dt=0.1):
    """"""
    n = int(np.round(duration / dt))
    x0 = np.zeros(n)
    x1 = np.zeros(n)
    for x, t in [(x0, t0), (x1, t1)]:
        x[np.array(np.round(t / dt), dtype=int)] = 1
    nk = int(np.round(3 * tau_vr / dt))
    if 2 * nk + 1 > n:
        nk = (n - 1) // 2
    T = np.arange(-nk, nk + 1) * dt
    kernel = np.exp(-T / tau_vr) / tau_vr
    for x in [x0, x1]:
        x[:] = np.convolve(x, kernel, "same")

    return np.sqrt(np.sum((x0 - x1) ** 2 * dt) / tau_vr)


def mean_vr_distance(
    network,
    neuron_spike_times,
    neuron_spike_idx,
    input_I,
    dt=0.1,
    disable_reporting=True,
):
    """"""
    duration = input_I.shape[1] * dt

    # Convert (t,i) format to list
    neuron_spikes, model_spikes = [], []
    for idx_repeat in tqdm(range(input_I.shape[0]), disable=disable_reporting):

        # "Real" neuron
        neuron_spikes.append(neuron_spike_times[neuron_spike_idx == idx_repeat])

        # Model
        m_spikes = network.run_and_readout_spikes(input_I[idx_repeat].reshape(1, -1))
        model_spikes.append(m_spikes)

    # Distance
    d = 0
    for t0, t1 in zip(neuron_spikes, model_spikes):
        d += van_rossum_distance(t0, t1, duration, dt=dt)
    d /= len(neuron_spikes)

    if disable_reporting is False:
        print("Mean vr distance: " + str(np.round(d, 2)))

    return d
