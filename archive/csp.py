import numpy as np
import matplotlib.pyplot as plt
import mne

from mne.decoding import CSP


class CSPAnalysis:

    # ============================================================
    # ORIGINAL DATASET EEG CHANNELS
    # ============================================================

    EEG_CHANNELS = [

        "EEG-Fz",

        "EEG-0",
        "EEG-1",
        "EEG-2",
        "EEG-3",
        "EEG-4",

        "EEG-5",
        "EEG-C3",
        "EEG-6",
        "EEG-Cz",
        "EEG-7",
        "EEG-C4",
        "EEG-8",

        "EEG-9",
        "EEG-10",
        "EEG-11",
        "EEG-12",
        "EEG-13",

        "EEG-14",
        "EEG-Pz",
        "EEG-15",
        "EEG-16"
    ]

    # ============================================================
    # STANDARD 10-20 CHANNEL NAMES
    # ============================================================

    STANDARD_CHANNELS = [

        "Fz",

        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",

        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",

        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",

        "P5",
        "Pz",
        "P3",
        "P1"
    ]

    # ============================================================
    # DATASET -> STANDARD MONTAGE MAPPING
    # ============================================================

    RENAME_DICT = {

        "EEG-Fz": "Fz",

        "EEG-0": "FC3",
        "EEG-1": "FC1",
        "EEG-2": "FCz",
        "EEG-3": "FC2",
        "EEG-4": "FC4",

        "EEG-5": "C5",
        "EEG-C3": "C3",
        "EEG-6": "C1",
        "EEG-Cz": "Cz",
        "EEG-7": "C2",
        "EEG-C4": "C4",
        "EEG-8": "C6",

        "EEG-9": "CP3",
        "EEG-10": "CP1",
        "EEG-11": "CPz",
        "EEG-12": "CP2",
        "EEG-13": "CP4",

        "EEG-14": "P5",
        "EEG-Pz": "Pz",
        "EEG-15": "P3",
        "EEG-16": "P1"
    }

    # ============================================================
    # PREPARE EEG EPOCHS
    # ============================================================

    @staticmethod
    def prepare_epochs(epochs):

        eeg_epochs = epochs.copy()

        # --------------------------------------------------------
        # Verify all expected EEG channels exist
        # --------------------------------------------------------

        missing = [
            ch
            for ch in CSPAnalysis.EEG_CHANNELS
            if ch not in eeg_epochs.ch_names
        ]

        if missing:

            raise ValueError(
                "Missing EEG channels: "
                f"{missing}"
            )

        # --------------------------------------------------------
        # Pick exactly the 22 EEG channels
        # --------------------------------------------------------

        eeg_epochs.pick(
            CSPAnalysis.EEG_CHANNELS
        )

        # --------------------------------------------------------
        # Rename channels to standard names
        # --------------------------------------------------------

        eeg_epochs.rename_channels(
            CSPAnalysis.RENAME_DICT
        )

        # --------------------------------------------------------
        # Apply standard montage
        # --------------------------------------------------------

        montage = (
            mne.channels.make_standard_montage(
                "standard_1005"
            )
        )

        eeg_epochs.set_montage(
            montage,
            on_missing="ignore"
        )

        # --------------------------------------------------------
        # Information
        # --------------------------------------------------------

        print()
        print("=" * 60)
        print("CSP EEG PREPARATION")
        print("=" * 60)

        print(
            "EEG channels:",
            len(eeg_epochs.ch_names)
        )

        print(
            "Expected EEG channels:",
            len(
                CSPAnalysis.STANDARD_CHANNELS
            )
        )

        print(
            "EOG channels removed:",
            len(
                epochs.ch_names
            )
            - len(
                eeg_epochs.ch_names
            )
        )

        print("\nChannel names:")

        for index, channel in enumerate(
            eeg_epochs.ch_names,
            start=1
        ):

            print(
                f"{index:2d}. {channel}"
            )

        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------

        if len(eeg_epochs.ch_names) != 22:

            raise ValueError(
                "CSP requires exactly 22 EEG channels, "
                f"but received "
                f"{len(eeg_epochs.ch_names)}"
            )

        return eeg_epochs

    # ============================================================
    # FIT CSP
    # ============================================================

    @staticmethod
    def fit_transform(
        epochs,
        conditions,
        n_components=4
    ):

        # --------------------------------------------------------
        # Prepare EEG
        # --------------------------------------------------------

        eeg_epochs = (
            CSPAnalysis.prepare_epochs(
                epochs
            )
        )

        # --------------------------------------------------------
        # Select requested classes
        # --------------------------------------------------------

        selected_epochs = (
            eeg_epochs[
                conditions
            ]
        )

        # --------------------------------------------------------
        # EEG data
        # --------------------------------------------------------

        X = selected_epochs.get_data(
            copy=True
        )

        # --------------------------------------------------------
        # Event IDs
        # --------------------------------------------------------

        first_id = (
            selected_epochs.event_id[
                conditions[0]
            ]
        )

        second_id = (
            selected_epochs.event_id[
                conditions[1]
            ]
        )

        # --------------------------------------------------------
        # Binary labels
        # --------------------------------------------------------

        y = np.where(
            selected_epochs.events[:, -1]
            == first_id,

            0,

            1
        )

        # --------------------------------------------------------
        # Safety checks
        # --------------------------------------------------------

        if X.shape[1] != 22:

            raise ValueError(
                "CSP received "
                f"{X.shape[1]} channels. "
                "Expected exactly 22."
            )

        if not np.all(
            np.isin(
                selected_epochs.events[:, -1],
                [first_id, second_id]
            )
        ):

            raise ValueError(
                "Unexpected event IDs detected "
                "during CSP preparation."
            )

        # --------------------------------------------------------
        # Create CSP
        # --------------------------------------------------------

        csp = CSP(
            n_components=n_components,
            reg=None,
            log=True,
            norm_trace=False
        )

        # --------------------------------------------------------
        # Fit CSP
        # --------------------------------------------------------

        X_csp = csp.fit_transform(
            X,
            y
        )

        # --------------------------------------------------------
        # Print information
        # --------------------------------------------------------

        print()
        print("=" * 60)
        print("CSP ANALYSIS")
        print("=" * 60)

        print(
            "Conditions:",
            conditions
        )

        print(
            "EEG data shape:",
            X.shape
        )

        print(
            "CSP feature shape:",
            X_csp.shape
        )

        print(
            f"{conditions[0]} trials:",
            np.sum(y == 0)
        )

        print(
            f"{conditions[1]} trials:",
            np.sum(y == 1)
        )

        return (
            X_csp,
            y,
            csp,
            selected_epochs
        )

    # ============================================================
    # PRINT CSP INFORMATION
    # ============================================================

    @staticmethod
    def print_components(csp):

        print()
        print("=" * 60)
        print("CSP COMPONENT INFORMATION")
        print("=" * 60)

        print(
            "Number of components:",
            csp.n_components
        )

        print(
            "Filters shape:",
            csp.filters_.shape
        )

        print(
            "Patterns shape:",
            csp.patterns_.shape
        )

    # ============================================================
    # CSP TOPOGRAPHIC PATTERNS
    # ============================================================

    @staticmethod
    def plot_patterns(
        csp,
        epochs,
        n_components=4,
        title="CSP Spatial Patterns"
    ):

        # --------------------------------------------------------
        # Create fresh Info
        # --------------------------------------------------------

        info = mne.create_info(
            ch_names=CSPAnalysis.STANDARD_CHANNELS,
            sfreq=epochs.info["sfreq"],
            ch_types="eeg"
        )

        # --------------------------------------------------------
        # Standard montage
        # --------------------------------------------------------

        montage = (
            mne.channels.make_standard_montage(
                "standard_1005"
            )
        )

        info.set_montage(
            montage
        )

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------

        fig, axes = plt.subplots(
            1,
            n_components,
            figsize=(16, 4)
        )

        if n_components == 1:
            axes = [axes]

        # --------------------------------------------------------
        # Plot patterns
        # --------------------------------------------------------

        for component in range(
            n_components
        ):

            pattern = (
                csp.patterns_[
                    :,
                    component
                ]
            )

            mne.viz.plot_topomap(
                pattern,
                info,
                axes=axes[component],
                show=False,
                contours=6
            )

            axes[
                component
            ].set_title(
                f"CSP Component "
                f"{component + 1}"
            )

        fig.suptitle(
            title,
            fontsize=16
        )

        plt.tight_layout()

        return fig

    # ============================================================
    # CSP FEATURE SPACE
    # ============================================================

    @staticmethod
    def plot_features(
        features,
        labels,
        class_names
    ):

        if features.shape[1] < 2:

            raise ValueError(
                "At least 2 CSP components "
                "are required to plot "
                "feature space."
            )

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        class_0 = (
            labels == 0
        )

        class_1 = (
            labels == 1
        )

        ax.scatter(
            features[class_0, 0],
            features[class_0, 1],
            label=class_names[0],
            alpha=0.7
        )

        ax.scatter(
            features[class_1, 0],
            features[class_1, 1],
            label=class_names[1],
            alpha=0.7
        )

        ax.set_xlabel(
            "CSP Component 1"
        )

        ax.set_ylabel(
            "CSP Component 2"
        )

        ax.set_title(
            f"CSP Feature Space - "
            f"{class_names[0]} vs "
            f"{class_names[1]}"
        )

        ax.legend()

        ax.grid(
            True
        )

        plt.tight_layout()

        return fig