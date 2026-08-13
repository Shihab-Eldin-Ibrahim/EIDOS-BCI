import numpy as np
import matplotlib.pyplot as plt
import csv

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mne.decoding import CSP

from core.loader import EEGLoader
from core.metadata import EEGMetadata

from preprocessing.epoching import EEGEpoching
from preprocessing.filtering import EEGFilter


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("../dataset/2a")

# Subjects A01 -> A09
SUBJECTS = [
    f"A{i:02d}"
    for i in range(1, 10)
]

# Motor imagery classes
CONDITIONS = [
    "left",
    "right",
    "feet",
    "tongue"
]

# Binary classification baseline
CLASSIFICATION_CONDITIONS = [
    "left",
    "right"
]

# Number of CSP components per frequency band
N_COMPONENTS = 4

# Cross-validation
N_SPLITS = 5

# Classification window
CLASSIFICATION_TMIN = 1.0
CLASSIFICATION_TMAX = 4.0

# Reproducibility
RANDOM_STATE = 42

# ============================================================
# FILTER BANK
# ============================================================
#
# Each frequency band gets its own CSP.
#
# Mu:
#   8-12 Hz
#
# Low beta:
#   12-16 Hz
#
# Mid beta:
#   16-20 Hz
#
# High beta:
#   20-24 Hz
#
# Broad beta:
#   24-30 Hz
#
# ============================================================

FILTER_BANKS = [
    ("8-12 Hz", 8, 12),
    ("12-16 Hz", 12, 16),
    ("16-20 Hz", 16, 20),
    ("20-24 Hz", 20, 24),
    ("24-30 Hz", 24, 30),
]

# Output files
BASELINE_RESULTS_FILE = (
    "results/multi_subject_baseline_results.csv"
)

FBCSP_RESULTS_FILE = (
    "results/multi_subject_fbcsp_results.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title):

    print("\n")
    print("-" * 60)
    print(title)
    print("-" * 60)


# ============================================================
# FIND DATASET FILES
# ============================================================

def find_subject_files():

    print_section(
        "SEARCHING FOR DATASET FILES"
    )

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{DATASET_DIR.resolve()}"
        )

    subject_files = {}

    for subject in SUBJECTS:

        file_path = (
            DATASET_DIR /
            f"{subject}T.gdf"
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

    if not subject_files:

        raise FileNotFoundError(
            "No training GDF files were found."
        )

    print(
        f"\nFound "
        f"{len(subject_files)} / "
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
    Prepare EEG data for classification.

    Returns
    -------
    X : ndarray
        Shape:
        (trials, channels, samples)

    y : ndarray
        Integer class labels.

    class_names : list
        Class names in label order.
    """

    print(
        "Conditions:",
        conditions
    )

    # --------------------------------------------------------
    # Select conditions
    # --------------------------------------------------------

    selected_epochs = epochs[
        list(conditions)
    ].copy()

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

    print(
        "EOG channels removed:",
        len(eog_channels)
    )

    # --------------------------------------------------------
    # Verify EEG channels
    # --------------------------------------------------------

    print(
        "EEG channels used:",
        len(selected_epochs.ch_names)
    )

    if len(selected_epochs.ch_names) != 22:

        raise ValueError(
            f"Expected 22 EEG channels, "
            f"but found "
            f"{len(selected_epochs.ch_names)}."
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
            ]
            .get_data(
                copy=True
            )
        )

        X_parts.append(
            condition_data
        )

        labels.extend(
            [class_index] *
            len(condition_data)
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    X = np.concatenate(
        X_parts,
        axis=0
    )

    y = np.asarray(
        labels,
        dtype=int
    )

    print(
        "\nEEG data shape:",
        X.shape
    )

    print(
        "Labels shape:",
        y.shape
    )

    print(
        "Total trials:",
        len(X)
    )

    print(
        f"Time window: "
        f"{tmin} - {tmax} seconds"
    )

    print(
        "Channels:",
        X.shape[1]
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print(
        "\nClass distribution:"
    )

    for class_index, condition in enumerate(
        conditions
    ):

        count = np.sum(
            y == class_index
        )

        print(
            f"{condition}: {count}"
        )

    return (
        X,
        y,
        list(conditions)
    )


# ============================================================
# STANDARD CSP CROSS-VALIDATION
# ============================================================

def run_csp_cross_validation(
    X,
    y,
    n_splits=5,
    n_components=4
):
    """
    Leakage-safe standard CSP classification.

    CSP is fitted separately inside every
    cross-validation fold.
    """

    print_section(
        "STANDARD CSP CROSS-VALIDATION"
    )

    classifiers = {

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

    results = {}

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # ========================================================
    # CLASSIFIERS
    # ========================================================

    for classifier_name, classifier in (
        classifiers.items()
    ):

        print_subsection(
            classifier_name
        )

        fold_accuracies = []

        for fold_number, (
            train_index,
            test_index
        ) in enumerate(
            cv.split(X, y),
            start=1
        ):

            X_train = X[
                train_index
            ]

            X_test = X[
                test_index
            ]

            y_train = y[
                train_index
            ]

            y_test = y[
                test_index
            ]

            print(
                f"\nFold {fold_number}"
            )

            # ------------------------------------------------
            # Fit CSP ONLY on training data
            # ------------------------------------------------

            csp = CSP(
                n_components=n_components,
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

            # ------------------------------------------------
            # Train classifier
            # ------------------------------------------------

            classifier.fit(
                X_train_csp,
                y_train
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            y_pred = (
                classifier.predict(
                    X_test_csp
                )
            )

            accuracy = (
                accuracy_score(
                    y_test,
                    y_pred
                )
                * 100
            )

            fold_accuracies.append(
                accuracy
            )

            print(
                f"Accuracy: "
                f"{accuracy:.2f}%"
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        fold_accuracies = np.asarray(
            fold_accuracies,
            dtype=float
        )

        mean_accuracy = np.mean(
            fold_accuracies
        )

        std_accuracy = np.std(
            fold_accuracies,
            ddof=1
        )

        minimum_accuracy = np.min(
            fold_accuracies
        )

        maximum_accuracy = np.max(
            fold_accuracies
        )

        print(
            f"\nMean: "
            f"{mean_accuracy:.2f}%"
        )

        print(
            f"Std: "
            f"{std_accuracy:.2f}%"
        )

        results[
            classifier_name
        ] = {

            "fold_accuracies":
                fold_accuracies,

            "mean":
                mean_accuracy,

            "std":
                std_accuracy,

            "min":
                minimum_accuracy,

            "max":
                maximum_accuracy
        }

    return results


# ============================================================
# FILTER-BANK CREATION
# ============================================================

def create_filter_bank(
    epochs
):
    """
    Create independently filtered versions
    of the epochs for FBCSP.
    """

    print_section(
        "CREATING FILTER BANK"
    )

    filter_bank = {}

    for band_name, low_freq, high_freq in FILTER_BANKS:

        print(
            f"Filtering {band_name}"
        )

        filtered = epochs.copy()

        filtered.filter(
            l_freq=low_freq,
            h_freq=high_freq,
            method="iir",
            verbose=False
        )

        filter_bank[
            band_name
        ] = filtered

    return filter_bank


# ============================================================
# PREPARE FBCSP DATA
# ============================================================

def prepare_fbcsp_data(
    epochs,
    conditions,
    tmin=1.0,
    tmax=4.0
):
    """
    Prepare one EEG dataset for every
    frequency band.

    Returns
    -------
    band_data : dict

        {
            "8-12 Hz": X,
            "12-16 Hz": X,
            ...
        }

    y : ndarray
        Class labels.
    """

    print_section(
        "PREPARING FBCSP DATA"
    )

    filter_bank = create_filter_bank(
        epochs
    )

    band_data = {}

    y = None

    for band_name, filtered_epochs in (
        filter_bank.items()
    ):

        print_subsection(
            f"Preparing {band_name}"
        )

        X, current_y, _ = (
            prepare_classification_data(
                filtered_epochs,
                conditions,
                tmin=tmin,
                tmax=tmax
            )
        )

        band_data[
            band_name
        ] = X

        if y is None:

            y = current_y

        else:

            if not np.array_equal(
                y,
                current_y
            ):

                raise ValueError(
                    "Labels differ between "
                    "frequency bands."
                )

    return (
        band_data,
        y
    )


# ============================================================
# FBCSP CROSS-VALIDATION
# ============================================================

def run_fbcsp_cross_validation(
    band_data,
    y,
    n_splits=5,
    n_components=4
):
    """
    Leakage-safe Filter-Bank CSP.

    For every CV fold:

    1. Split training/testing data.
    2. Fit one CSP per frequency band
       using training data only.
    3. Transform training and testing data.
    4. Concatenate CSP features.
    5. Train classifier.
    6. Evaluate on unseen test data.
    """

    print_section(
        "FILTER-BANK CSP CROSS-VALIDATION"
    )

    classifiers = {

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

    results = {}

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    band_names = list(
        band_data.keys()
    )

    # ========================================================
    # CLASSIFIER LOOP
    # ========================================================

    for classifier_name, classifier in (
        classifiers.items()
    ):

        print_subsection(
            f"FBCSP + {classifier_name}"
        )

        fold_accuracies = []

        for fold_number, (
            train_index,
            test_index
        ) in enumerate(
            cv.split(
                band_data[
                    band_names[0]
                ],
                y
            ),
            start=1
        ):

            print(
                f"\nFold {fold_number}"
            )

            train_features = []
            test_features = []

            # =================================================
            # PROCESS EACH FREQUENCY BAND
            # =================================================

            for band_name in band_names:

                X = band_data[
                    band_name
                ]

                X_train = X[
                    train_index
                ]

                X_test = X[
                    test_index
                ]

                y_train = y[
                    train_index
                ]

                # -------------------------------------------------
                # CSP is fitted ONLY on training data
                # -------------------------------------------------

                csp = CSP(
                    n_components=n_components,
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

                train_features.append(
                    X_train_csp
                )

                test_features.append(
                    X_test_csp
                )

                print(
                    f"{band_name}: "
                    f"{X_train_csp.shape}"
                )

            # =================================================
            # CONCATENATE ALL FREQUENCY FEATURES
            # =================================================

            X_train_fbcsp = np.concatenate(
                train_features,
                axis=1
            )

            X_test_fbcsp = np.concatenate(
                test_features,
                axis=1
            )

            print(
                "Combined training features:",
                X_train_fbcsp.shape
            )

            print(
                "Combined testing features:",
                X_test_fbcsp.shape
            )

            # =================================================
            # CLASSIFIER
            # =================================================

            classifier.fit(
                X_train_fbcsp,
                y_train
            )

            y_test = y[
                test_index
            ]

            y_pred = (
                classifier.predict(
                    X_test_fbcsp
                )
            )

            accuracy = (
                accuracy_score(
                    y_test,
                    y_pred
                )
                * 100
            )

            fold_accuracies.append(
                accuracy
            )

            print(
                f"Accuracy: "
                f"{accuracy:.2f}%"
            )

        # =====================================================
        # STATISTICS
        # =====================================================

        fold_accuracies = np.asarray(
            fold_accuracies,
            dtype=float
        )

        mean_accuracy = np.mean(
            fold_accuracies
        )

        std_accuracy = np.std(
            fold_accuracies,
            ddof=1
        )

        minimum_accuracy = np.min(
            fold_accuracies
        )

        maximum_accuracy = np.max(
            fold_accuracies
        )

        print(
            f"\nMean Accuracy: "
            f"{mean_accuracy:.2f}%"
        )

        print(
            f"Standard Deviation: "
            f"{std_accuracy:.2f}%"
        )

        print(
            f"Minimum: "
            f"{minimum_accuracy:.2f}%"
        )

        print(
            f"Maximum: "
            f"{maximum_accuracy:.2f}%"
        )

        results[
            classifier_name
        ] = {

            "fold_accuracies":
                fold_accuracies,

            "mean":
                mean_accuracy,

            "std":
                std_accuracy,

            "min":
                minimum_accuracy,

            "max":
                maximum_accuracy
        }

    return results


# ============================================================
# PRINT CLASSIFIER COMPARISON
# ============================================================

def print_classifier_comparison(
    results,
    title="CLASSIFIER COMPARISON"
):

    print("\n")
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"{'Classifier':<30}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
        f"{'Min':>12}"
        f"{'Max':>12}"
    )

    print("-" * 90)

    for classifier_name, data in (
        results.items()
    ):

        print(
            f"{classifier_name:<30}"
            f"{data['mean']:>10.2f}%"
            f"{data['std']:>10.2f}%"
            f"{data['min']:>10.2f}%"
            f"{data['max']:>10.2f}%"
        )

    print("=" * 90)


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(
    subject,
    file_path
):

    print_section(
        f"SUBJECT {subject}"
    )

    print(
        f"Dataset: {file_path}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    loader = EEGLoader()

    raw = loader.load(
        str(file_path)
    )

    # ========================================================
    # METADATA
    # ========================================================

    EEGMetadata.print_summary(
        raw
    )

    # ========================================================
    # EPOCHING
    # ========================================================

    print_section(
        f"{subject} - MOTOR IMAGERY EPOCHING"
    )

    epochs = (
        EEGEpoching
        .create_motor_imagery_epochs(
            raw
        )
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

    # ========================================================
    # EOG REMOVAL
    # ========================================================

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

    # ========================================================
    # STANDARD 8-30 Hz FILTER
    # ========================================================

    print_section(
        f"{subject} - STANDARD FILTERING"
    )

    filtered_epochs = (
        EEGFilter.bandpass(
            epochs
        )
    )

    print(
        "Band-pass filter: 8-30 Hz"
    )

    # ========================================================
    # CLASSIFICATION DATA
    # ========================================================

    print_section(
        f"{subject} - STANDARD CSP DATA"
    )

    X, y, _ = (
        prepare_classification_data(
            filtered_epochs,
            CLASSIFICATION_CONDITIONS,
            tmin=CLASSIFICATION_TMIN,
            tmax=CLASSIFICATION_TMAX
        )
    )

    # ========================================================
    # STANDARD CSP
    # ========================================================

    baseline_results = (
        run_csp_cross_validation(
            X,
            y,
            n_splits=N_SPLITS,
            n_components=N_COMPONENTS
        )
    )

    print_classifier_comparison(
        baseline_results,
        title=(
            f"{subject} - "
            "STANDARD CSP RESULTS"
        )
    )

    # ========================================================
    # FBCSP
    # ========================================================

    print_section(
        f"{subject} - FBCSP"
    )

    band_data, fbcsp_y = (
        prepare_fbcsp_data(
            epochs,
            CLASSIFICATION_CONDITIONS,
            tmin=CLASSIFICATION_TMIN,
            tmax=CLASSIFICATION_TMAX
        )
    )

    # Verify labels
    if not np.array_equal(
        y,
        fbcsp_y
    ):

        raise ValueError(
            "Standard CSP and FBCSP "
            "labels do not match."
        )

    fbcsp_results = (
        run_fbcsp_cross_validation(
            band_data,
            fbcsp_y,
            n_splits=N_SPLITS,
            n_components=N_COMPONENTS
        )
    )

    print_classifier_comparison(
        fbcsp_results,
        title=(
            f"{subject} - "
            "FBCSP RESULTS"
        )
    )

    return {
        "baseline": baseline_results,
        "fbcsp": fbcsp_results
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    all_results,
    output_file,
    experiment_type
):

    output_path = Path(
        output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Subject",
            "Experiment",
            "Classifier",
            "Mean Accuracy (%)",
            "Std (%)",
            "Minimum (%)",
            "Maximum (%)"
        ])

        for subject, subject_results in (
            all_results.items()
        ):

            results = subject_results[
                experiment_type
            ]

            for classifier, data in (
                results.items()
            ):

                writer.writerow([
                    subject,
                    experiment_type,
                    classifier,
                    f"{data['mean']:.4f}",
                    f"{data['std']:.4f}",
                    f"{data['min']:.4f}",
                    f"{data['max']:.4f}"
                ])

    print(
        f"\nResults saved to:"
        f"\n{output_path}"
    )


# ============================================================
# MULTI-SUBJECT SUMMARY
# ============================================================

def print_multi_subject_summary(
    all_results,
    experiment_type
):

    print("\n")
    print("=" * 100)

    if experiment_type == "baseline":

        print(
            "MULTI-SUBJECT STANDARD CSP RESULTS"
        )

    else:

        print(
            "MULTI-SUBJECT FBCSP RESULTS"
        )

    print("=" * 100)

    classifier_names = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        f"{'Subject':<12}"
        f"{'LDA':>15}"
        f"{'SVM':>15}"
        f"{'Logistic':>20}"
    )

    print("-" * 100)

    # --------------------------------------------------------
    # Subject results
    # --------------------------------------------------------

    for subject, subject_results in (
        all_results.items()
    ):

        results = subject_results[
            experiment_type
        ]

        print(
            f"{subject:<12}"
            f"{results['LDA']['mean']:>13.2f}%"
            f"{results['SVM']['mean']:>13.2f}%"
            f"{results['Logistic Regression']['mean']:>18.2f}%"
        )

    print("-" * 100)

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    print(
        "\nOVERALL SUBJECT STATISTICS"
    )

    classifier_overall_means = {}

    for classifier in classifier_names:

        subject_means = np.asarray([
            all_results[
                subject
            ][
                experiment_type
            ][
                classifier
            ][
                "mean"
            ]

            for subject in all_results
        ])

        overall_mean = np.mean(
            subject_means
        )

        overall_std = np.std(
            subject_means,
            ddof=1
        )

        minimum = np.min(
            subject_means
        )

        maximum = np.max(
            subject_means
        )

        classifier_overall_means[
            classifier
        ] = overall_mean

        best_subject_index = np.argmax(
            subject_means
        )

        worst_subject_index = np.argmin(
            subject_means
        )

        subjects = list(
            all_results.keys()
        )

        print(
            f"\n{classifier}"
        )

        print(
            f"Mean across subjects: "
            f"{overall_mean:.2f}%"
        )

        print(
            f"Std across subjects: "
            f"{overall_std:.2f}%"
        )

        print(
            f"Best subject: "
            f"{subjects[best_subject_index]} "
            f"({maximum:.2f}%)"
        )

        print(
            f"Worst subject: "
            f"{subjects[worst_subject_index]} "
            f"({minimum:.2f}%)"
        )

    # --------------------------------------------------------
    # Best classifier
    # --------------------------------------------------------

    best_classifier = max(
        classifier_overall_means,
        key=classifier_overall_means.get
    )

    print("\n")

    print(
        f"Best overall classifier: "
        f"{best_classifier}"
    )

    print(
        f"Overall mean accuracy: "
        f"{classifier_overall_means[best_classifier]:.2f}%"
    )

    print("=" * 100)


# ============================================================
# PLOT MULTI-SUBJECT RESULTS
# ============================================================

def plot_multi_subject_results(
    all_results,
    experiment_type
):

    subjects = list(
        all_results.keys()
    )

    classifier_names = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    x = np.arange(
        len(subjects)
    )

    width = 0.25

    for index, classifier in enumerate(
        classifier_names
    ):

        values = [
            all_results[
                subject
            ][
                experiment_type
            ][
                classifier
            ][
                "mean"
            ]

            for subject in subjects
        ]

        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=classifier
        )

        for bar, value in zip(
            bars,
            values
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                bar.get_height()
                + 1,

                f"{value:.1f}%",

                ha="center",
                va="bottom",

                fontsize=8
            )

    if experiment_type == "baseline":

        title = (
            "Multi-Subject Standard CSP "
            "Motor Imagery Classification"
        )

    else:

        title = (
            "Multi-Subject FBCSP "
            "Motor Imagery Classification"
        )

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Subject",
        fontsize=14
    )

    ax.set_ylabel(
        "Mean Accuracy (%)",
        fontsize=14
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        subjects
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_yticks(
        np.arange(
            0,
            101,
            10
        )
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    return fig


# ============================================================
# COMPARE BASELINE VS FBCSP
# ============================================================

def print_baseline_vs_fbcsp(
    all_results
):

    print("\n")
    print("=" * 100)
    print(
        "STANDARD CSP vs FBCSP"
    )
    print("=" * 100)

    classifiers = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    for classifier in classifiers:

        baseline_values = [
            all_results[
                subject
            ][
                "baseline"
            ][
                classifier
            ][
                "mean"
            ]

            for subject in all_results
        ]

        fbcsp_values = [
            all_results[
                subject
            ][
                "fbcsp"
            ][
                classifier
            ][
                "mean"
            ]

            for subject in all_results
        ]

        baseline_mean = np.mean(
            baseline_values
        )

        fbcsp_mean = np.mean(
            fbcsp_values
        )

        improvement = (
            fbcsp_mean
            - baseline_mean
        )

        print(
            f"\n{classifier}"
        )

        print(
            f"Standard CSP: "
            f"{baseline_mean:.2f}%"
        )

        print(
            f"FBCSP: "
            f"{fbcsp_mean:.2f}%"
        )

        print(
            f"Improvement: "
            f"{improvement:+.2f} percentage points"
        )

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 100)
    print("EIDOS-BCI")
    print("FILTER-BANK CSP EXPERIMENT")
    print("=" * 100)

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
        ", ".join(
            CLASSIFICATION_CONDITIONS
        )
    )

    print(
        "\nClassification window:",
        f"{CLASSIFICATION_TMIN}-"
        f"{CLASSIFICATION_TMAX} seconds"
    )

    print(
        "\nCSP components per band:",
        N_COMPONENTS
    )

    print(
        "\nCross-validation:",
        f"{N_SPLITS}-fold StratifiedKFold"
    )

    print(
        "\nFilter bank:"
    )

    for band_name, low, high in FILTER_BANKS:

        print(
            f"  {band_name}"
        )

    # ========================================================
    # FIND SUBJECT FILES
    # ========================================================

    subject_files = (
        find_subject_files()
    )

    # ========================================================
    # PROCESS SUBJECTS
    # ========================================================

    all_results = {}

    for subject, file_path in (
        subject_files.items()
    ):

        try:

            results = process_subject(
                subject,
                file_path
            )

            all_results[
                subject
            ] = results

            print(
                f"\n✓ {subject} "
                f"completed successfully."
            )

        except Exception as error:

            print(
                f"\n✗ ERROR processing "
                f"{subject}:"
            )

            print(
                repr(error)
            )

            print(
                "\nContinuing with next subject..."
            )

    # ========================================================
    # CHECK
    # ========================================================

    if not all_results:

        raise RuntimeError(
            "No subjects were successfully processed."
        )

    # ========================================================
    # STANDARD CSP SUMMARY
    # ========================================================

    print_multi_subject_summary(
        all_results,
        "baseline"
    )

    # ========================================================
    # FBCSP SUMMARY
    # ========================================================

    print_multi_subject_summary(
        all_results,
        "fbcsp"
    )

    # ========================================================
    # BASELINE VS FBCSP
    # ========================================================

    print_baseline_vs_fbcsp(
        all_results
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_results(
        all_results,
        BASELINE_RESULTS_FILE,
        "baseline"
    )

    save_results(
        all_results,
        FBCSP_RESULTS_FILE,
        "fbcsp"
    )

    # ========================================================
    # PLOTS
    # ========================================================

    print_section(
        "GENERATING RESULT PLOTS"
    )

    plot_multi_subject_results(
        all_results,
        "baseline"
    )

    plot_multi_subject_results(
        all_results,
        "fbcsp"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)

    print(
        f"\nSubjects successfully processed: "
        f"{len(all_results)} / "
        f"{len(SUBJECTS)}"
    )

    print(
        f"\nBaseline results:"
        f"\n{BASELINE_RESULTS_FILE}"
    )

    print(
        f"\nFBCSP results:"
        f"\n{FBCSP_RESULTS_FILE}"
    )

    print(
        "\nNext research stages:"
    )

    print(
        "1. Analyze FBCSP improvement"
    )

    print(
        "2. Confusion matrices"
    )

    print(
        "3. Statistical significance testing"
    )

    print(
        "4. Four-class classification"
    )

    print(
        "5. Subject-independent classification"
    )

    print(
        "6. Paper experiment tables"
    )

    print("=" * 100)

    plt.show()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()