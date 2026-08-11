import matplotlib.pyplot as plt

from core.loader import EEGLoader
from core.metadata import EEGMetadata

from preprocessing.epoching import EEGEpoching
from preprocessing.filtering import EEGFilter

from analysis.erd_ers import ERDERS
from analysis.csp import CSPAnalysis


def main():

    # ============================================================
    # LOAD EEG DATA
    # ============================================================

    print("\n")
    print("=" * 60)
    print("EIDOS-BCI")
    print("=" * 60)

    loader = EEGLoader()

    raw = loader.load(
        "dataset/2a/A01T.gdf"
    )

    # ============================================================
    # EEG METADATA
    # ============================================================

    EEGMetadata.print_summary(
        raw
    )

    # ============================================================
    # CREATE MOTOR IMAGERY EPOCHS
    # ============================================================

    print("\n")
    print("=" * 60)
    print("MOTOR IMAGERY EPOCHING")
    print("=" * 60)

    epochs = EEGEpoching.create_motor_imagery_epochs(
        raw
    )

    print(
        "\nNumber of epochs:",
        len(epochs)
    )

    print(
        "Epoch shape:",
        epochs.get_data().shape
    )

    print(
        "Sampling rate:",
        epochs.info["sfreq"],
        "Hz"
    )

    print(
        "Channels:",
        len(epochs.ch_names)
    )

    print(
        "Time range:",
        epochs.tmin,
        "to",
        epochs.tmax,
        "seconds"
    )

    # ============================================================
    # EPOCH COUNTS
    # ============================================================

    print("\nEpoch counts:")

    for condition in [
        "left",
        "right",
        "feet",
        "tongue"
    ]:

        print(
            condition,
            ":",
            len(epochs[condition])
        )

    # ============================================================
    # FILTER EEG
    # ============================================================

    print("\n")
    print("=" * 60)
    print("EEG FILTERING")
    print("=" * 60)

    filtered_epochs = EEGFilter.bandpass(
        epochs
    )

    print(
        "Band-pass filter:",
        "8-30 Hz"
    )

    # ============================================================
    # PSD ANALYSIS
    # ============================================================

    print("\n")
    print("=" * 60)
    print("PSD ANALYSIS")
    print("=" * 60)

    psd_figures = []

    for condition in [
        "left",
        "right",
        "feet",
        "tongue"
    ]:

        print(
            f"Calculating PSD for {condition}"
        )

        fig = filtered_epochs[
            condition
        ].compute_psd(
            fmin=8,
            fmax=30
        ).plot(
            average=True,
            picks=[
                "EEG-C3",
                "EEG-Cz",
                "EEG-C4"
            ],
            show=False
        )

        fig.suptitle(
            f"Motor Imagery PSD - "
            f"{condition.capitalize()}",
            fontsize=16
        )

        psd_figures.append(
            fig
        )

    # ============================================================
    # ERD / ERS ANALYSIS
    # ============================================================

    print("\n")
    print("=" * 60)
    print("ERD / ERS ANALYSIS")
    print("=" * 60)

    # ------------------------------------------------------------
    # Motor rhythm frequency bands
    # ------------------------------------------------------------

    mu_band = (
        8,
        13
    )

    beta_band = (
        13,
        30
    )

    erd_ers_results = {}

    for condition in [
        "left",
        "right",
        "feet",
        "tongue"
    ]:

        print(
            f"\nAnalyzing {condition} imagery"
        )

        condition_epochs = epochs[
            condition
        ]

        # --------------------------------------------------------
        # MU BAND
        # --------------------------------------------------------

        mu_power = ERDERS.calculate_band_power(
            condition_epochs,
            mu_band[0],
            mu_band[1]
        )

        # --------------------------------------------------------
        # BETA BAND
        # --------------------------------------------------------

        beta_power = ERDERS.calculate_band_power(
            condition_epochs,
            beta_band[0],
            beta_band[1]
        )

        # --------------------------------------------------------
        # ERD / ERS
        # --------------------------------------------------------

        mu_erd_ers = ERDERS.calculate_erd_ers(
            mu_power,
            condition_epochs
        )

        beta_erd_ers = ERDERS.calculate_erd_ers(
            beta_power,
            condition_epochs
        )

        erd_ers_results[
            condition
        ] = {

            "mu": mu_erd_ers,

            "beta": beta_erd_ers
        }

        print(
            "Mu ERD/ERS shape:",
            mu_erd_ers.shape
        )

        print(
            "Beta ERD/ERS shape:",
            beta_erd_ers.shape
        )

    # ============================================================
    # ERD / ERS VISUALIZATION
    # ============================================================

    print("\n")
    print("=" * 60)
    print("ERD / ERS VISUALIZATION")
    print("=" * 60)

    erd_fig, erd_axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    erd_axes = erd_axes.flatten()

    for index, condition in enumerate([
        "left",
        "right",
        "feet",
        "tongue"
    ]):

        ERDERS.plot_erd_ers(
            erd_ers_results[
                condition
            ]["mu"],
            epochs[condition],
            [
                "EEG-C3",
                "EEG-Cz",
                "EEG-C4"
            ],
            f"Mu ERD/ERS - "
            f"{condition.capitalize()}",
            erd_axes[index]
        )

    erd_fig.suptitle(
        "Motor Imagery Mu-Band ERD/ERS",
        fontsize=18
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ]
    )

    # ============================================================
    # CSP ANALYSIS
    # ============================================================

    print("\n")
    print("=" * 60)
    print("CSP ANALYSIS")
    print("=" * 60)

    # ------------------------------------------------------------
    # CSP will classify:
    #
    # LEFT HAND vs RIGHT HAND
    # ------------------------------------------------------------

    (
        csp_features,
        csp_labels,
        csp,
        csp_epochs
    ) = CSPAnalysis.fit_transform(

        epochs,

        conditions=[
            "left",
            "right"
        ],

        n_components=4
    )

    # ============================================================
    # CSP COMPONENT INFORMATION
    # ============================================================

    CSPAnalysis.print_components(
        csp
    )

    # ============================================================
    # CSP SPATIAL PATTERNS
    # ============================================================

    print("\n")
    print("=" * 60)
    print("CSP SPATIAL PATTERNS")
    print("=" * 60)

    CSPAnalysis.plot_patterns(
        csp,
        csp_epochs,
        n_components=4,
        title=(
            "CSP Spatial Patterns - "
            "Left vs Right Motor Imagery"
        )
    )

    # ============================================================
    # CSP FEATURE SPACE
    # ============================================================

    print("\n")
    print("=" * 60)
    print("CSP FEATURE SPACE")
    print("=" * 60)

    CSPAnalysis.plot_features(
        csp_features,
        csp_labels,
        [
            "Left Hand",
            "Right Hand"
        ]
    )

    # ============================================================
    # RAW EEG VISUALIZATION
    # ============================================================

    print("\nShowing RAW EEG")

    epochs[
        "left"
    ].plot(
        n_epochs=1,
        n_channels=22,
        scalings="auto",
        block=False
    )

    # ============================================================
    # FILTERED EEG VISUALIZATION
    # ============================================================

    print(
        "Showing FILTERED EEG"
    )

    filtered_epochs[
        "left"
    ].plot(
        n_epochs=1,
        n_channels=22,
        scalings="auto",
        block=False
    )

    # ============================================================
    # SHOW ALL PLOTS
    # ============================================================

    print("\n")
    print("=" * 60)
    print("DISPLAYING ALL PLOTS")
    print("=" * 60)

    plt.show()


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()