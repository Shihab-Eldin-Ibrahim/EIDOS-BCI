import mne


class EEGFilter:

    @staticmethod
    def bandpass(epochs, low_freq=8.0, high_freq=30.0):
        """
        Apply an 8-30 Hz band-pass filter to EEG epochs.
        """

        filtered_epochs = epochs.copy()

        filtered_epochs.filter(
            l_freq=low_freq,
            h_freq=high_freq,
            picks="eeg"
        )

        return filtered_epochs