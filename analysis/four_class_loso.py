import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from mne.decoding import CSP

from core.loader import EEGLoader
from core.metadata import EEGMetadata

from preprocessing.epoching import EEGEpoching
from preprocessing.filtering import EEGFilter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset" / "2a"

RESULTS_DIR = PROJECT_ROOT / "results" / "four_class" / "cross_subject"
print("\nProject root:", PROJECT_ROOT)
print("Dataset directory:", DATASET_DIR)
print("Results directory:", RESULTS_DIR)
STANDARD_RESULTS_FILE = (
    RESULTS_DIR / "standard_csp_loso_results.csv"
)

FBCSP_RESULTS_FILE = (
    RESULTS_DIR / "fbcsp_loso_results.csv"
)

COMPARISON_RESULTS_FILE = (
    RESULTS_DIR / "loso_comparison.csv"
)

SUBJECT_RESULTS_FILE = (
    RESULTS_DIR / "subject_results.csv"
)

CONFUSION_MATRIX_DIR = (
    RESULTS_DIR / "confusion_matrices"
)

PLOTS_DIR = (
    RESULTS_DIR / "plots"
)

COMPARISON_PLOT = (
    PLOTS_DIR / "loso_comparison.png"
)

SUBJECT_PLOT = (
    PLOTS_DIR / "subject_performance.png"
)


# ============================================================
# SUBJECTS
# ============================================================

SUBJECTS = [
    f"A{i:02d}"
    for i in range(1, 10)
]


# ============================================================
# FOUR MOTOR-IMAGERY CLASSES
# ============================================================

CONDITIONS = [
    "left",
    "right",
    "feet",
    "tongue"
]


# ============================================================
# CLASSIFICATION PARAMETERS
# ============================================================

CLASSIFICATION_TMIN = 1.0
CLASSIFICATION_TMAX = 4.0

N_COMPONENTS = 4


# ============================================================
# FBCSP FILTER BANK
# ============================================================

FILTER_BANKS = [
    ("8-12 Hz", 8, 12),
    ("12-16 Hz", 12, 16),
    ("16-20 Hz", 16, 20),
    ("20-24 Hz", 20, 24),
    ("24-30 Hz", 24, 30),
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_STATE = 42


# ============================================================
# PRINT HELPERS
# ============================================================

def print_section(title):

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_subsection(title):

    print("\n")
    print("-" * 80)
    print(title)
    print("-" * 80)


# ============================================================
# FIND DATASET FILES
# ============================================================

def find_subject_files():

    print_section("SEARCHING FOR DATASET FILES")

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{DATASET_DIR}"
        )

    subject_files = {}

    for subject in SUBJECTS:

        file_path = (
            DATASET_DIR
            / f"{subject}T.gdf"
        )

        if file_path.exists():

            subject_files[subject] = file_path

            print(
                f"[FOUND] {file_path}"
            )

        else:

            print(
                f"[MISSING] {file_path}"
            )

    if len(subject_files) != len(SUBJECTS):

        raise FileNotFoundError(
            "All nine subjects are required "
            "for LOSO evaluation."
        )

    print(
        f"\nFound {len(subject_files)} / "
        f"{len(SUBJECTS)} subjects."
    )

    return subject_files


# ============================================================
# PREPARE CLASSIFICATION DATA
# ============================================================

def prepare_classification_data(
    epochs,
    conditions,
    tmin=1.0,
    tmax=4.0
):
    """
    Convert MNE epochs into X and y.

    X:
        trials x channels x samples

    y:
        integer class labels
    """

    selected_epochs = (
        epochs[list(conditions)].copy()
    )

    # --------------------------------------------------------
    # Remove EOG channels
    # --------------------------------------------------------

    eog_channels = [
        ch
        for ch in selected_epochs.ch_names
        if "EOG" in ch.upper()
    ]

    if eog_channels:

        selected_epochs.drop_channels(
            eog_channels
        )

    # --------------------------------------------------------
    # Verify EEG channels
    # --------------------------------------------------------

    eeg_channels = [
        ch
        for ch in selected_epochs.ch_names
        if selected_epochs.get_channel_types(
            picks=[ch]
        )[0] == "eeg"
    ]

    if len(eeg_channels) != 22:

        raise ValueError(
            f"Expected 22 EEG channels, "
            f"found {len(eeg_channels)}."
        )

    # --------------------------------------------------------
    # Crop classification window
    # --------------------------------------------------------

    selected_epochs.crop(
        tmin=tmin,
        tmax=tmax,
        include_tmax=False
    )

    # --------------------------------------------------------
    # Build X and y
    # --------------------------------------------------------

    X_parts = []
    labels = []

    for class_index, condition in enumerate(
        conditions
    ):

        condition_data = (
            selected_epochs[
                condition
            ].get_data(copy=True)
        )

        X_parts.append(
            condition_data
        )

        labels.extend(
            [class_index]
            * len(condition_data)
        )

    X = np.concatenate(
        X_parts,
        axis=0
    )

    y = np.asarray(
        labels,
        dtype=int
    )

    return X, y


# ============================================================
# CREATE CLASSIFIERS
# ============================================================

def create_classifiers():

    return {

        "LDA":
            LinearDiscriminantAnalysis(),

        "SVM":
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale"
            ),

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            )
    }


# ============================================================
# LOAD AND PREPROCESS ONE SUBJECT
# ============================================================

def load_subject_data(
    subject,
    file_path
):

    print_subsection(
        f"LOADING {subject}"
    )

    print(
        f"File: {file_path}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    loader = EEGLoader()

    raw = loader.load(
        str(file_path)
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    EEGMetadata.print_summary(
        raw
    )

    # --------------------------------------------------------
    # Epoching
    # --------------------------------------------------------

    epochs = (
        EEGEpoching
        .create_motor_imagery_epochs(
            raw
        )
    )

    print(
        f"\nEpochs: {len(epochs)}"
    )

    # --------------------------------------------------------
    # Remove EOG
    # --------------------------------------------------------

    eog_channels = [
        ch
        for ch in epochs.ch_names
        if "EOG" in ch.upper()
    ]

    if eog_channels:

        epochs = epochs.copy()

        epochs.drop_channels(
            eog_channels
        )

        print(
            "Removed EOG channels:",
            eog_channels
        )

    # --------------------------------------------------------
    # Standard 8-30 Hz filtering
    # --------------------------------------------------------

    filtered_epochs = (
        EEGFilter.bandpass(
            epochs
        )
    )

    # --------------------------------------------------------
    # Standard CSP data
    # --------------------------------------------------------

    X, y = prepare_classification_data(
        filtered_epochs,
        CONDITIONS,
        CLASSIFICATION_TMIN,
        CLASSIFICATION_TMAX
    )

    # --------------------------------------------------------
    # FBCSP data
    # --------------------------------------------------------

    fbcsp_data = {}

    for band_name, low_freq, high_freq in FILTER_BANKS:

        print(
            f"{subject}: preparing {band_name}"
        )

        band_epochs = epochs.copy()

        band_epochs.filter(
            l_freq=low_freq,
            h_freq=high_freq,
            picks="eeg",
            verbose=False
        )

        band_X, band_y = (
            prepare_classification_data(
                band_epochs,
                CONDITIONS,
                CLASSIFICATION_TMIN,
                CLASSIFICATION_TMAX
            )
        )

        if not np.array_equal(
            y,
            band_y
        ):

            raise ValueError(
                f"Labels differ for {subject} "
                f"in {band_name}."
            )

        fbcsp_data[
            band_name
        ] = band_X

    return {
        "standard_X": X,
        "y": y,
        "fbcsp_data": fbcsp_data
    }


# ============================================================
# FIT STANDARD CSP ON TRAINING SUBJECTS
# ============================================================

def run_standard_csp_loso_fold(
    train_data,
    test_data,
    test_subject
):

    print_section(
        f"STANDARD CSP - TEST SUBJECT {test_subject}"
    )

    classifiers = create_classifiers()

    results = {}

    X_train = np.concatenate(
        [
            train_data[s]["standard_X"]
            for s in train_data
        ],
        axis=0
    )

    y_train = np.concatenate(
        [
            train_data[s]["y"]
            for s in train_data
        ],
        axis=0
    )

    X_test = test_data["standard_X"]
    y_test = test_data["y"]

    print(
        "Training data:",
        X_train.shape
    )

    print(
        "Testing data:",
        X_test.shape
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # CSP is fitted ONLY on training subjects.
    # --------------------------------------------------------

    csp = CSP(
        n_components=N_COMPONENTS,
        reg=None,
        log=True,
        norm_trace=False
    )

    X_train_csp = (
        csp.fit_transform(
            X_train,
            y_train
        )
    )

    X_test_csp = (
        csp.transform(
            X_test
        )
    )

    print(
        "CSP training features:",
        X_train_csp.shape
    )

    print(
        "CSP testing features:",
        X_test_csp.shape
    )

    for classifier_name, classifier in (
        classifiers.items()
    ):

        classifier.fit(
            X_train_csp,
            y_train
        )

        y_pred = classifier.predict(
            X_test_csp
        )

        accuracy = (
            accuracy_score(
                y_test,
                y_pred
            ) * 100
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=np.arange(
                len(CONDITIONS)
            )
        )

        results[
            classifier_name
        ] = {
            "accuracy": accuracy,
            "confusion_matrix": cm
        }

        print(
            f"{classifier_name}: "
            f"{accuracy:.2f}%"
        )

    return results


# ============================================================
# FIT FBCSP ON TRAINING SUBJECTS
# ============================================================

def run_fbcsp_loso_fold(
    train_data,
    test_data,
    test_subject
):

    print_section(
        f"FBCSP - TEST SUBJECT {test_subject}"
    )

    classifiers = create_classifiers()

    train_features = []
    test_features = []

    y_train = np.concatenate(
        [
            train_data[s]["y"]
            for s in train_data
        ],
        axis=0
    )

    y_test = test_data["y"]

    # --------------------------------------------------------
    # Process every frequency band
    # --------------------------------------------------------

    for band_name, _, _ in FILTER_BANKS:

        print_subsection(
            band_name
        )

        X_train_band = np.concatenate(
            [
                train_data[s][
                    "fbcsp_data"
                ][band_name]
                for s in train_data
            ],
            axis=0
        )

        X_test_band = (
            test_data[
                "fbcsp_data"
            ][band_name]
        )

        print(
            "Training band data:",
            X_train_band.shape
        )

        print(
            "Testing band data:",
            X_test_band.shape
        )

        # ----------------------------------------------------
        # CSP fitted ONLY on training subjects
        # ----------------------------------------------------

        csp = CSP(
            n_components=N_COMPONENTS,
            reg=None,
            log=True,
            norm_trace=False
        )

        X_train_csp = (
            csp.fit_transform(
                X_train_band,
                y_train
            )
        )

        X_test_csp = (
            csp.transform(
                X_test_band
            )
        )

        train_features.append(
            X_train_csp
        )

        test_features.append(
            X_test_csp
        )

    # --------------------------------------------------------
    # Combine all frequency-band features
    # --------------------------------------------------------

    X_train_fbcsp = np.concatenate(
        train_features,
        axis=1
    )

    X_test_fbcsp = np.concatenate(
        test_features,
        axis=1
    )

    print(
        "\nCombined training features:",
        X_train_fbcsp.shape
    )

    print(
        "Combined testing features:",
        X_test_fbcsp.shape
    )

    results = {}

    for classifier_name, classifier in (
        classifiers.items()
    ):

        classifier.fit(
            X_train_fbcsp,
            y_train
        )

        y_pred = classifier.predict(
            X_test_fbcsp
        )

        accuracy = (
            accuracy_score(
                y_test,
                y_pred
            ) * 100
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=np.arange(
                len(CONDITIONS)
            )
        )

        results[
            classifier_name
        ] = {
            "accuracy": accuracy,
            "confusion_matrix": cm
        }

        print(
            f"{classifier_name}: "
            f"{accuracy:.2f}%"
        )

    return results


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    cm,
    subject,
    experiment_type,
    classifier_name
):

    CONFUSION_MATRIX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"{subject}_"
        f"{experiment_type}_"
        f"{classifier_name.replace(' ', '_')}.png"
    )

    output_file = (
        CONFUSION_MATRIX_DIR
        / filename
    )

    cm_percent = (
        cm.astype(float)
        /
        cm.sum(
            axis=1,
            keepdims=True
        )
        * 100
    )

    cm_percent = np.nan_to_num(
        cm_percent
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm_percent,
        display_labels=CONDITIONS
    )

    display.plot(
        ax=ax,
        values_format=".1f",
        cmap="Blues",
        colorbar=True
    )

    ax.set_title(
        f"{subject} - "
        f"{experiment_type.upper()} - "
        f"{classifier_name}\n"
        "LOSO Four-Class Motor Imagery"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SAVE STANDARD CSP RESULTS
# ============================================================

def save_standard_results(
    results
):

    STANDARD_RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        STANDARD_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Test Subject",
            "Classifier",
            "Accuracy (%)"
        ])

        for subject in results:

            for classifier in results[
                subject
            ]:

                writer.writerow([
                    subject,
                    classifier,
                    f"{results[subject][classifier]['accuracy']:.4f}"
                ])


# ============================================================
# SAVE FBCSP RESULTS
# ============================================================

def save_fbcsp_results(
    results
):

    FBCSP_RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        FBCSP_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Test Subject",
            "Classifier",
            "Accuracy (%)"
        ])

        for subject in results:

            for classifier in results[
                subject
            ]:

                writer.writerow([
                    subject,
                    classifier,
                    f"{results[subject][classifier]['accuracy']:.4f}"
                ])


# ============================================================
# SAVE COMPARISON
# ============================================================

def save_comparison(
    standard_results,
    fbcsp_results
):

    classifiers = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    with open(
        COMPARISON_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Classifier",
            "Standard CSP Mean (%)",
            "FBCSP Mean (%)",
            "Improvement (pp)"
        ])

        for classifier in classifiers:

            standard_values = [
                standard_results[
                    subject
                ][classifier]["accuracy"]
                for subject in SUBJECTS
            ]

            fbcsp_values = [
                fbcsp_results[
                    subject
                ][classifier]["accuracy"]
                for subject in SUBJECTS
            ]

            standard_mean = np.mean(
                standard_values
            )

            fbcsp_mean = np.mean(
                fbcsp_values
            )

            improvement = (
                fbcsp_mean
                - standard_mean
            )

            writer.writerow([
                classifier,
                f"{standard_mean:.4f}",
                f"{fbcsp_mean:.4f}",
                f"{improvement:.4f}"
            ])


# ============================================================
# SAVE SUBJECT RESULTS
# ============================================================

def save_subject_results(
    standard_results,
    fbcsp_results
):

    with open(
        SUBJECT_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Test Subject",

            "CSP LDA",
            "FBCSP LDA",

            "CSP SVM",
            "FBCSP SVM",

            "CSP Logistic Regression",
            "FBCSP Logistic Regression"
        ])

        for subject in SUBJECTS:

            writer.writerow([
                subject,

                f"{standard_results[subject]['LDA']['accuracy']:.4f}",
                f"{fbcsp_results[subject]['LDA']['accuracy']:.4f}",

                f"{standard_results[subject]['SVM']['accuracy']:.4f}",
                f"{fbcsp_results[subject]['SVM']['accuracy']:.4f}",

                f"{standard_results[subject]['Logistic Regression']['accuracy']:.4f}",
                f"{fbcsp_results[subject]['Logistic Regression']['accuracy']:.4f}"
            ])


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_summary(
    standard_results,
    fbcsp_results
):

    classifiers = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    print_section(
        "LOSO FINAL RESULTS"
    )

    print(
        f"{'Classifier':<25}"
        f"{'CSP':>15}"
        f"{'FBCSP':>15}"
        f"{'Change':>15}"
    )

    print("-" * 75)

    for classifier in classifiers:

        standard_values = np.array([
            standard_results[
                subject
            ][classifier]["accuracy"]
            for subject in SUBJECTS
        ])

        fbcsp_values = np.array([
            fbcsp_results[
                subject
            ][classifier]["accuracy"]
            for subject in SUBJECTS
        ])

        standard_mean = np.mean(
            standard_values
        )

        fbcsp_mean = np.mean(
            fbcsp_values
        )

        change = (
            fbcsp_mean
            - standard_mean
        )

        print(
            f"{classifier:<25}"
            f"{standard_mean:>13.2f}%"
            f"{fbcsp_mean:>13.2f}%"
            f"{change:>+13.2f}"
        )

    print("-" * 75)


# ============================================================
# PLOT COMPARISON
# ============================================================

def plot_comparison(
    standard_results,
    fbcsp_results
):

    classifiers = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    standard_values = []
    fbcsp_values = []

    for classifier in classifiers:

        standard_values.append(
            np.mean([
                standard_results[
                    subject
                ][classifier]["accuracy"]
                for subject in SUBJECTS
            ])
        )

        fbcsp_values.append(
            np.mean([
                fbcsp_results[
                    subject
                ][classifier]["accuracy"]
                for subject in SUBJECTS
            ])
        )

    x = np.arange(
        len(classifiers)
    )

    width = 0.35

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars1 = ax.bar(
        x - width / 2,
        standard_values,
        width,
        label="Standard CSP"
    )

    bars2 = ax.bar(
        x + width / 2,
        fbcsp_values,
        width,
        label="FBCSP"
    )

    for bar, value in zip(
        bars1,
        standard_values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center"
        )

    for bar, value in zip(
        bars2,
        fbcsp_values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center"
        )

    ax.set_title(
        "LOSO Four-Class Motor Imagery:\n"
        "Standard CSP vs FBCSP"
    )

    ax.set_xlabel(
        "Classifier"
    )

    ax.set_ylabel(
        "Mean Test Accuracy (%)"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        classifiers
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        COMPARISON_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SUBJECT PERFORMANCE PLOT
# ============================================================

def plot_subject_performance(
    standard_results,
    fbcsp_results
):

    classifier = "LDA"

    standard_values = [
        standard_results[
            subject
        ][classifier]["accuracy"]
        for subject in SUBJECTS
    ]

    fbcsp_values = [
        fbcsp_results[
            subject
        ][classifier]["accuracy"]
        for subject in SUBJECTS
    ]

    x = np.arange(
        len(SUBJECTS)
    )

    width = 0.35

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        x - width / 2,
        standard_values,
        width,
        label="Standard CSP"
    )

    ax.bar(
        x + width / 2,
        fbcsp_values,
        width,
        label="FBCSP"
    )

    ax.set_title(
        "LOSO Subject Performance - LDA"
    )

    ax.set_xlabel(
        "Held-Out Subject"
    )

    ax.set_ylabel(
        "Test Accuracy (%)"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        SUBJECTS
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        SUBJECT_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "EIDOS-BCI"
    )

    print(
        "FOUR-CLASS SUBJECT-INDEPENDENT "
        "LOSO EXPERIMENT"
    )

    print(
        "\nDataset:"
        "\nBCI Competition IV Dataset 2a"
    )

    print(
        "\nSubjects:",
        ", ".join(SUBJECTS)
    )

    print(
        "\nClasses:",
        ", ".join(CONDITIONS)
    )

    print(
        "\nClassification window:",
        f"{CLASSIFICATION_TMIN}-"
        f"{CLASSIFICATION_TMAX} seconds"
    )

    print(
        "\nCSP components:",
        N_COMPONENTS
    )

    print(
        "\nFilter bank:"
    )

    for band_name, _, _ in FILTER_BANKS:

        print(
            f"  {band_name}"
        )

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "\nResults directory:",
        RESULTS_DIR
    )

    # ========================================================
    # FIND DATASET
    # ========================================================

    subject_files = (
        find_subject_files()
    )

    # ========================================================
    # LOAD ALL SUBJECTS
    # ========================================================

    all_data = {}

    print_section(
        "PREPROCESSING ALL SUBJECTS"
    )

    for subject in SUBJECTS:

        all_data[
            subject
        ] = load_subject_data(
            subject,
            subject_files[subject]
        )

        print(
            f"\n✓ {subject} prepared."
        )

    # ========================================================
    # LOSO
    # ========================================================

    standard_results = {}
    fbcsp_results = {}

    for test_subject in SUBJECTS:

        print_section(
            f"LOSO FOLD: TEST = {test_subject}"
        )

        train_subjects = [
            subject
            for subject in SUBJECTS
            if subject != test_subject
        ]

        print(
            "Training subjects:",
            ", ".join(train_subjects)
        )

        print(
            "Test subject:",
            test_subject
        )

        train_data = {
            subject:
                all_data[subject]
            for subject in train_subjects
        }

        test_data = (
            all_data[test_subject]
        )

        # ----------------------------------------------------
        # Standard CSP
        # ----------------------------------------------------

        standard_results[
            test_subject
        ] = run_standard_csp_loso_fold(
            train_data,
            test_data,
            test_subject
        )

        # ----------------------------------------------------
        # FBCSP
        # ----------------------------------------------------

        fbcsp_results[
            test_subject
        ] = run_fbcsp_loso_fold(
            train_data,
            test_data,
            test_subject
        )

        # ----------------------------------------------------
        # Save confusion matrices
        # ----------------------------------------------------

        for classifier in standard_results[
            test_subject
        ]:

            save_confusion_matrix(
                standard_results[
                    test_subject
                ][classifier][
                    "confusion_matrix"
                ],
                test_subject,
                "standard_csp",
                classifier
            )

        for classifier in fbcsp_results[
            test_subject
        ]:

            save_confusion_matrix(
                fbcsp_results[
                    test_subject
                ][classifier][
                    "confusion_matrix"
                ],
                test_subject,
                "fbcsp",
                classifier
            )

    # ========================================================
    # SAVE
    # ========================================================

    save_standard_results(
        standard_results
    )

    save_fbcsp_results(
        fbcsp_results
    )

    save_comparison(
        standard_results,
        fbcsp_results
    )

    save_subject_results(
        standard_results,
        fbcsp_results
    )

    # ========================================================
    # PLOTS
    # ========================================================

    plot_comparison(
        standard_results,
        fbcsp_results
    )

    plot_subject_performance(
        standard_results,
        fbcsp_results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        standard_results,
        fbcsp_results
    )

    print_section(
        "LOSO EXPERIMENT COMPLETE"
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {STANDARD_RESULTS_FILE}"
    )

    print(
        f"2. {FBCSP_RESULTS_FILE}"
    )

    print(
        f"3. {COMPARISON_RESULTS_FILE}"
    )

    print(
        f"4. {SUBJECT_RESULTS_FILE}"
    )

    print(
        f"5. {COMPARISON_PLOT}"
    )

    print(
        f"6. {SUBJECT_PLOT}"
    )

    print(
        f"7. {CONFUSION_MATRIX_DIR}"
    )

    print(
        "\nNext checklist stage:"
    )

    print(
        "→ Paper experiment tables"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()