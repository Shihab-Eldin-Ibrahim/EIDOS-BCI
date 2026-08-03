"""
metadata.py

Utilities for displaying EEG dataset information.
"""

class EEGMetadata:

    @staticmethod
    def print_summary(raw):

        print("\n" + "=" * 50)
        print("EEG DATASET INFORMATION")
        print("=" * 50)

        print(f"Channels           : {len(raw.ch_names)}")
        print(f"Sampling Frequency : {raw.info['sfreq']} Hz")
        print(f"Duration           : {raw.times[-1]:.2f} seconds")
        print(f"Samples            : {raw.n_times}")

        print("\nChannel Names:")

        for i, name in enumerate(raw.ch_names, start=1):
            print(f"{i:2d}. {name}")

        print("=" * 50)