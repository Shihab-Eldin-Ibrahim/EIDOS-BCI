import numpy as np
import matplotlib.pyplot as plt
import csv

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
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

RESULTS_DIR = PROJECT_ROOT / "results" / "four_class"

BASELINE_RESULTS_FILE = (
    RESULTS_DIR / "four_class_standard_csp_results.csv"
)

FBCSP_RESULTS_FILE = (
    RESULTS_DIR / "four_class_fbcsp_results.csv"
)

COMPARISON_RESULTS_FILE = (
    RESULTS_DIR / "four_class_comparison.csv"
)

SUBJECT_RESULTS_FILE = (
    RESULTS_DIR / "four_class_subject_results.csv"
)

CONFUSION_MATRIX_DIR = (
    RESULTS_DIR / "confusion_matrices"
)

PLOT_FILE = (
    RESULTS_DIR / "four_class_comparison.png"
)


# Subjects A01 -> A09

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


# Classification window

CLASSIFICATION_TMIN = 1.0
CLASSIFICATION_TMAX = 4.0


# CSP components per frequency band

N_COMPONENTS = 4


# Cross-validation

N_SPLITS = 5


# Reproducibility

RANDOM_STATE = 42


# ============================================================
# FILTER BANK
# ============================================================

FILTER_BANKS = [
    ("8-12 Hz", 8, 12),
    ("12-16 Hz", 12, 16),
    ("16-20 Hz", 16, 20),
    ("20-24 Hz", 20, 24),
    ("24-30 Hz", 24, 30),
]


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

    print_section(
        "SEARCHING FOR DATASET FILES"
    )

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{DATASET_DIR}"
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
    Prepare EEG data for four-class classification.

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
    # Select four classes
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
# STANDARD CSP FOUR-CLASS CLASSIFICATION
# ============================================================

def run_csp_cross_validation(
    X,
    y,
    n_splits=5,
    n_components=4,
    subject=None
):
    """
    Leakage-safe four-class Standard CSP.

    CSP is fitted separately inside every
    cross-validation fold.
    """

    print_section(
        "FOUR-CLASS STANDARD CSP CROSS-VALIDATION"
    )

    classifiers = create_classifiers()

    results = {}

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for classifier_name, classifier in (
        classifiers.items()
    ):

        print_subsection(
            classifier_name
        )

        fold_accuracies = []

        all_y_true = []
        all_y_pred = []

        for fold_number, (
            train_index,
            test_index
        ) in enumerate(
            cv.split(X, y),
            start=1
        ):

            print(
                f"\nFold {fold_number}"
            )

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

            # ------------------------------------------------
            # CSP fitted ONLY on training data
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

            all_y_true.extend(
                y_test
            )

            all_y_pred.extend(
                y_pred
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

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            all_y_true,
            all_y_pred,
            labels=np.arange(
                len(CONDITIONS)
            )
        )

        print(
            f"\nMean: "
            f"{mean_accuracy:.2f}%"
        )

        print(
            f"Std: "
            f"{std_accuracy:.2f}%"
        )

        print(
            f"Min: "
            f"{minimum_accuracy:.2f}%"
        )

        print(
            f"Max: "
            f"{maximum_accuracy:.2f}%"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(cm)

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
                maximum_accuracy,

            "confusion_matrix":
                cm
        }

        # ----------------------------------------------------
        # Save confusion matrix
        # ----------------------------------------------------

        if subject is not None:

            save_confusion_matrix(
                cm,
                classifier_name,
                "standard_csp",
                subject
            )

    return results


# ============================================================
# CREATE FILTER BANK
# ============================================================

def create_filter_bank(epochs):

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
# FBCSP FOUR-CLASS CLASSIFICATION
# ============================================================

def run_fbcsp_cross_validation(
    band_data,
    y,
    n_splits=5,
    n_components=4,
    subject=None
):
    """
    Leakage-safe four-class FBCSP.

    For every CV fold:

    1. Split training/testing data.
    2. Fit one CSP per frequency band.
    3. Transform training/testing data.
    4. Concatenate CSP features.
    5. Train classifier.
    6. Evaluate on unseen test data.
    """

    print_section(
        "FOUR-CLASS FILTER-BANK CSP CROSS-VALIDATION"
    )

    classifiers = create_classifiers()

    results = {}

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    band_names = list(
        band_data.keys()
    )

    for classifier_name, classifier in (
        classifiers.items()
    ):

        print_subsection(
            f"FBCSP + {classifier_name}"
        )

        fold_accuracies = []

        all_y_true = []
        all_y_pred = []

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

            y_train = y[
                train_index
            ]

            y_test = y[
                test_index
            ]

            # ------------------------------------------------
            # Process each frequency band
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Concatenate features
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Classifier
            # ------------------------------------------------

            classifier.fit(
                X_train_fbcsp,
                y_train
            )

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

            all_y_true.extend(
                y_test
            )

            all_y_pred.extend(
                y_pred
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

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            all_y_true,
            all_y_pred,
            labels=np.arange(
                len(CONDITIONS)
            )
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

        print(
            "\nConfusion Matrix:"
        )

        print(cm)

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
                maximum_accuracy,

            "confusion_matrix":
                cm
        }

        # ----------------------------------------------------
        # Save confusion matrix
        # ----------------------------------------------------

        if subject is not None:

            save_confusion_matrix(
                cm,
                classifier_name,
                "fbcsp",
                subject
            )

    return results


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    cm,
    classifier_name,
    experiment_type,
    subject
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
        CONFUSION_MATRIX_DIR /
        filename
    )

    # Normalize rows to percentages

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
        "Four-Class Motor Imagery"
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

    print(
        f"Confusion matrix saved:"
        f"\n{output_file}"
    )


# ============================================================
# PRINT CLASSIFIER COMPARISON
# ============================================================

def print_classifier_comparison(
    results,
    title
):

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'Classifier':<30}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
        f"{'Min':>12}"
        f"{'Max':>12}"
    )

    print("-" * 100)

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

    print("=" * 100)


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
    # REMOVE EOG
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
        f"{subject} - STANDARD 8-30 Hz FILTERING"
    )

    filtered_epochs = (
        EEGFilter.bandpass(
            epochs
        )
    )

    # ========================================================
    # FOUR-CLASS DATA
    # ========================================================

    print_section(
        f"{subject} - FOUR-CLASS STANDARD CSP DATA"
    )

    X, y, class_names = (
        prepare_classification_data(
            filtered_epochs,
            CONDITIONS,
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
            n_components=N_COMPONENTS,
            subject=subject
        )
    )

    print_classifier_comparison(
        baseline_results,
        title=(
            f"{subject} - "
            "FOUR-CLASS STANDARD CSP RESULTS"
        )
    )

    # ========================================================
    # FBCSP
    # ========================================================

    print_section(
        f"{subject} - FOUR-CLASS FBCSP"
    )

    band_data, fbcsp_y = (
        prepare_fbcsp_data(
            epochs,
            CONDITIONS,
            tmin=CLASSIFICATION_TMIN,
            tmax=CLASSIFICATION_TMAX
        )
    )

    # ========================================================
    # VERIFY LABELS
    # ========================================================

    if not np.array_equal(
        y,
        fbcsp_y
    ):

        raise ValueError(
            "Standard CSP and FBCSP "
            "labels do not match."
        )

    # ========================================================
    # RUN FBCSP
    # ========================================================

    fbcsp_results = (
        run_fbcsp_cross_validation(
            band_data,
            fbcsp_y,
            n_splits=N_SPLITS,
            n_components=N_COMPONENTS,
            subject=subject
        )
    )

    print_classifier_comparison(
        fbcsp_results,
        title=(
            f"{subject} - "
            "FOUR-CLASS FBCSP RESULTS"
        )
    )

    return {

        "baseline":
            baseline_results,

        "fbcsp":
            fbcsp_results
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    all_results,
    output_file,
    experiment_type
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
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
        f"\n{output_file}"
    )


# ============================================================
# SAVE SUBJECT COMPARISON
# ============================================================

def save_subject_comparison(
    all_results
):

    SUBJECT_RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SUBJECT_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Subject",
            "Classifier",
            "Standard CSP (%)",
            "FBCSP (%)",
            "Improvement (pp)"
        ])

        classifiers = [
            "LDA",
            "SVM",
            "Logistic Regression"
        ]

        for subject in all_results:

            for classifier in classifiers:

                baseline = (
                    all_results[
                        subject
                    ][
                        "baseline"
                    ][
                        classifier
                    ][
                        "mean"
                    ]
                )

                fbcsp = (
                    all_results[
                        subject
                    ][
                        "fbcsp"
                    ][
                        classifier
                    ][
                        "mean"
                    ]
                )

                improvement = (
                    fbcsp -
                    baseline
                )

                writer.writerow([
                    subject,
                    classifier,
                    f"{baseline:.4f}",
                    f"{fbcsp:.4f}",
                    f"{improvement:.4f}"
                ])

    print(
        f"\nSubject comparison saved to:"
        f"\n{SUBJECT_RESULTS_FILE}"
    )


# ============================================================
# MULTI-SUBJECT SUMMARY
# ============================================================

def print_multi_subject_summary(
    all_results,
    experiment_type
):

    print("\n")
    print("=" * 110)

    if experiment_type == "baseline":

        print(
            "FOUR-CLASS MULTI-SUBJECT STANDARD CSP RESULTS"
        )

    else:

        print(
            "FOUR-CLASS MULTI-SUBJECT FBCSP RESULTS"
        )

    print("=" * 110)

    classifier_names = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    print(
        f"{'Subject':<12}"
        f"{'LDA':>15}"
        f"{'SVM':>15}"
        f"{'Logistic':>20}"
    )

    print("-" * 110)

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

    print("-" * 110)

    print(
        "\nOVERALL SUBJECT STATISTICS"
    )

    overall_means = {}

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

        overall_means[
            classifier
        ] = overall_mean

        subjects = list(
            all_results.keys()
        )

        best_index = np.argmax(
            subject_means
        )

        worst_index = np.argmin(
            subject_means
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
            f"{subjects[best_index]} "
            f"({maximum:.2f}%)"
        )

        print(
            f"Worst subject: "
            f"{subjects[worst_index]} "
            f"({minimum:.2f}%)"
        )

    best_classifier = max(
        overall_means,
        key=overall_means.get
    )

    print("\n")

    print(
        f"Best overall classifier: "
        f"{best_classifier}"
    )

    print(
        f"Overall mean accuracy: "
        f"{overall_means[best_classifier]:.2f}%"
    )

    print("=" * 110)


# ============================================================
# STANDARD CSP VS FBCSP
# ============================================================

def print_baseline_vs_fbcsp(
    all_results
):

    print("\n")
    print("=" * 110)
    print(
        "FOUR-CLASS STANDARD CSP vs FBCSP"
    )
    print("=" * 110)

    classifiers = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    for classifier in classifiers:

        baseline_values = np.asarray([

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

        ])

        fbcsp_values = np.asarray([

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

        ])

        baseline_mean = np.mean(
            baseline_values
        )

        fbcsp_mean = np.mean(
            fbcsp_values
        )

        improvement = (
            fbcsp_mean -
            baseline_mean
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
            f"{improvement:+.2f} "
            f"percentage points"
        )

    print("=" * 110)


# ============================================================
# SAVE COMPARISON
# ============================================================

def save_comparison(
    all_results
):

    COMPARISON_RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Classifier",
            "Standard CSP Mean (%)",
            "FBCSP Mean (%)",
            "Change (pp)"
        ])

        for classifier in classifiers:

            baseline = np.mean([

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

            ])

            fbcsp = np.mean([

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

            ])

            change = (
                fbcsp -
                baseline
            )

            writer.writerow([
                classifier,
                f"{baseline:.4f}",
                f"{fbcsp:.4f}",
                f"{change:.4f}"
            ])

    print(
        f"\nComparison saved to:"
        f"\n{COMPARISON_RESULTS_FILE}"
    )


# ============================================================
# PLOT RESULTS
# ============================================================

def plot_comparison(
    all_results
):

    classifiers = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    baseline_values = []
    fbcsp_values = []

    for classifier in classifiers:

        baseline_values.append(
            np.mean([

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

            ])
        )

        fbcsp_values.append(
            np.mean([

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

            ])
        )

    x = np.arange(
        len(classifiers)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars1 = ax.bar(
        x - width / 2,
        baseline_values,
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
        baseline_values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom"
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
            ha="center",
            va="bottom"
        )

    ax.set_title(
        "Four-Class Motor Imagery:\n"
        "Standard CSP vs FBCSP",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Classifier"
    )

    ax.set_ylabel(
        "Mean Accuracy (%)"
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

    PLOT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        PLOT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nComparison plot saved to:"
        f"\n{PLOT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 110)
    print("EIDOS-BCI")
    print("FOUR-CLASS MOTOR IMAGERY EXPERIMENT")
    print("=" * 110)

    print(
        "\nDataset:"
        "\nBCI Competition IV Dataset 2a"
    )

    print(
        "\nSubjects:",
        ", ".join(SUBJECTS)
    )

    print(
        "\nClasses:"
    )

    for index, condition in enumerate(
        CONDITIONS
    ):

        print(
            f"  {index}: {condition}"
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

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "\nResults directory:",
        RESULTS_DIR
    )

    # ========================================================
    # FIND FILES
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
    # COMPARISON
    # ========================================================

    print_baseline_vs_fbcsp(
        all_results
    )

    # ========================================================
    # SAVE CSV FILES
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

    save_subject_comparison(
        all_results
    )

    save_comparison(
        all_results
    )

    # ========================================================
    # PLOT
    # ========================================================

    plot_comparison(
        all_results
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 110)
    print("FOUR-CLASS EXPERIMENT COMPLETE")
    print("=" * 110)

    print(
        f"\nSubjects successfully processed: "
        f"{len(all_results)} / "
        f"{len(SUBJECTS)}"
    )

    print(
        "\nClasses:"
    )

    print(
        "Left Hand"
    )

    print(
        "Right Hand"
    )

    print(
        "Feet"
    )

    print(
        "Tongue"
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {BASELINE_RESULTS_FILE}"
    )

    print(
        f"2. {FBCSP_RESULTS_FILE}"
    )

    print(
        f"3. {SUBJECT_RESULTS_FILE}"
    )

    print(
        f"4. {COMPARISON_RESULTS_FILE}"
    )

    print(
        f"5. {PLOT_FILE}"
    )

    print(
        f"6. Confusion matrices:"
        f"\n   {CONFUSION_MATRIX_DIR}"
    )

    print(
        "\nNext research stage:"
    )

    print(
        "1. Four-class statistical significance testing"
    )

    print(
        "2. Subject-independent classification"
    )

    print(
        "3. Cross-subject confusion matrices"
    )

    print(
        "4. Paper experiment tables"
    )

    print("=" * 110)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()