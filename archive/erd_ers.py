import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


class ERDERS:

    @staticmethod
    def calculate_band_power(epochs, fmin, fmax):

        filtered = epochs.copy().filter(
            l_freq=fmin,
            h_freq=fmax,
            picks="eeg",
            verbose=False
        )

        data = filtered.get_data()

        analytic_signal = np.abs(
            hilbert(data, axis=-1)
        )

        power = analytic_signal ** 2

        return power

    @staticmethod
    def calculate_erd_ers(power, epochs):

        times = epochs.times

        # Baseline: -1 to 0 seconds
        baseline_mask = (
            (times >= -1.0) &
            (times < 0.0)
        )

        baseline_power = np.mean(
            power[:, :, baseline_mask],
            axis=2,
            keepdims=True
        )

        erd_ers = (
            (power - baseline_power)
            / baseline_power
        ) * 100

        return erd_ers

    @staticmethod
    def smooth(data, window_size=25):

        """
        Smooth a 1D ERD/ERS signal using a moving average.
        """

        if window_size <= 1:
            return data

        kernel = np.ones(window_size) / window_size

        return np.convolve(
            data,
            kernel,
            mode="same"
        )

    @staticmethod
    def plot_erd_ers(
        erd_ers,
        epochs,
        channel_names,
        title,
        ax,
        smoothing=25
    ):

        times = epochs.times

        for channel in channel_names:

            channel_index = epochs.ch_names.index(channel)

            # Average across trials
            mean_erd_ers = np.mean(
                erd_ers[:, channel_index, :],
                axis=0
            )

            # Smooth only for visualization
            smoothed = ERDERS.smooth(
                mean_erd_ers,
                smoothing
            )

            ax.plot(
                times,
                smoothed,
                label=channel
            )

        # Motor imagery starts at 0 seconds
        ax.axvline(
            0,
            linestyle="--"
        )

        # Baseline reference
        ax.axhline(
            0,
            linestyle="--"
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("ERD / ERS (%)")
        ax.set_title(title)

        ax.legend()
        ax.grid(True)