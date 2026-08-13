import numpy as np
import matplotlib.pyplot as plt
import csv

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)

from mne.decoding import CSP

from core.loader import EEGLoader
from preprocessing.epoching import EEGEpoching
from preprocessing.filtering import EEGFilter


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset/2a")

SUBJECTS = [
    f"A{i:02d}"
    for i in range(1, 10)
]

# Binary motor imagery experiment
CONDITIONS = [
    "left",
    "right"
]

# Classification window
CLASSIFICATION_TMIN = 1.0
CLASSIFICATION_TMAX = 4.0

# EEG filter
FILTER_LOW = 8
FILTER_HIGH = 30

# CSP
N_COMPONENTS = 4

# Cross-validation
N_SPLITS = 5

# Reproducibility
RANDOM_STATE = 42

# Results
RESULTS_DIR = Path(
    "results/confusion_matrices"
)

STANDARD_DIR = (
    RESULTS_DIR / "standard_csp"
)

FBCSP_DIR = (
    RESULTS_DIR / "fbcsp"
)

AGGREGATE_DIR = (
    RESULTS_DIR / "aggregate"
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
# FIND SUBJECT FILES
# ============================================================

def find_subject_files():

    print_section(
        "SEARCHING FOR DATASET FILES"
    )

    subject_files = {}

    for subject in SUBJECTS:

        file_path = (
            DATASET_DIR /
            f"{subject}T.gdf"
        )

        if file_path.exists():

            subject_files[
                subject
            ] = file_path

            print(
                f"[FOUND] {file_path}"
            )

        else:

            print(
                f"[MISSING] {file_path}"
            )

    if not subject_files:

        raise FileNotFoundError(
            "No subject GDF files found."
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
    Prepare EEG data for binary classification.

    Returns
    -------
    X : ndarray
        Shape:
        (trials, channels, samples)

    y : ndarray
        Integer labels.

    class_names : list
        Class names.
    """

    print(
        "Conditions:",
        conditions
    )

    # --------------------------------------------------------
    # Select classes
    # --------------------------------------------------------

    selected_epochs = (
        epochs[
            list(conditions)
        ].copy()
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

    print(
        "EOG channels removed:",
        len(eog_channels)
    )

    # --------------------------------------------------------
    # Verify EEG channels
    # --------------------------------------------------------

    eeg_channel_count = (
        len(
            selected_epochs.ch_names
        )
    )

    print(
        "EEG channels:",
        eeg_channel_count
    )

    if eeg_channel_count != 22:

        raise ValueError(
            f"Expected 22 EEG channels, "
            f"found {eeg_channel_count}."
        )

    # --------------------------------------------------------
    # Crop
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

        data = (
            selected_epochs[
                condition
            ].get_data(
                copy=True
            )
        )

        X_parts.append(
            data
        )

        labels.extend(
            [class_index] *
            len(data)
        )

    X = np.concatenate(
        X_parts,
        axis=0
    )

    y = np.asarray(
        labels,
        dtype=int
    )

    print(
        "Classification data:",
        X.shape
    )

    print(
        "Labels:",
        y.shape
    )

    return (
        X,
        y,
        list(conditions)
    )


# ============================================================
# CLASSIFIERS
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
# STANDARD CSP
# ============================================================

def run_standard_csp_cv(
    X,
    y,
    n_splits=5,
    n_components=4
):
    """
    Leakage-safe Standard CSP.

    CSP is fitted independently
    inside every CV fold.

    Returns
    -------
    results : dict
        Accuracy and predictions for
        every classifier.
    """

    print_section(
        "STANDARD CSP CROSS-VALIDATION"
    )

    classifiers = (
        create_classifiers()
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = {}

    for classifier_name, classifier in (
        classifiers.items()
    ):

        print_subsection(
            f"Standard CSP - {classifier_name}"
        )

        y_true_all = []
        y_pred_all = []

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
                f"Fold {fold_number}"
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
            # Predict
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
                ) * 100
            )

            fold_accuracies.append(
                accuracy
            )

            # ------------------------------------------------
            # Store predictions
            # ------------------------------------------------

            y_true_all.extend(
                y_test.tolist()
            )

            y_pred_all.extend(
                y_pred.tolist()
            )

            print(
                f"Accuracy: "
                f"{accuracy:.2f}%"
            )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        y_true_all = np.asarray(
            y_true_all
        )

        y_pred_all = np.asarray(
            y_pred_all
        )

        cm = confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=[
                0,
                1
            ]
        )

        normalized_cm = (
            cm.astype(float)
            /
            cm.sum(
                axis=1,
                keepdims=True
            )
        )

        overall_accuracy = (
            accuracy_score(
                y_true_all,
                y_pred_all
            ) * 100
        )

        results[
            classifier_name
        ] = {

            "fold_accuracies":
                np.asarray(
                    fold_accuracies
                ),

            "mean_accuracy":
                np.mean(
                    fold_accuracies
                ),

            "overall_accuracy":
                overall_accuracy,

            "y_true":
                y_true_all,

            "y_pred":
                y_pred_all,

            "confusion_matrix":
                cm,

            "normalized_confusion_matrix":
                normalized_cm
        }

        print(
            f"\nOverall accuracy: "
            f"{overall_accuracy:.2f}%"
        )

        print(
            "\nConfusion matrix:"
        )

        print(cm)

    return results


# ============================================================
# FBCSP
# ============================================================

def run_fbcsp_cv(
    X,
    y,
    n_splits=5
):
    """
    Leakage-safe Filter-Bank CSP.

    Frequency bands:
        8-12 Hz
        12-16 Hz
        16-20 Hz
        20-24 Hz
        24-28 Hz

    CSP is independently fitted
    inside every fold and every
    frequency band.
    """

    print_section(
        "FBCSP CROSS-VALIDATION"
    )

    # --------------------------------------------------------
    # Frequency bands
    # --------------------------------------------------------

    bands = [
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 24),
        (24, 28)
    ]

    classifiers = (
        create_classifiers()
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = {}

    for classifier_name, classifier in (
        classifiers.items()
    ):

        print_subsection(
            f"FBCSP - {classifier_name}"
        )

        y_true_all = []
        y_pred_all = []

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

            # =================================================
            # FILTER BANK
            # =================================================

            train_features = []
            test_features = []

            for low, high in bands:

                print(
                    f"Band: "
                    f"{low}-{high} Hz"
                )

                # ------------------------------------------------
                # Filter TRAINING DATA
                # ------------------------------------------------

                train_band = (
                    mne.filter.filter_data(
                        X_train,
                        sfreq=250,
                        l_freq=low,
                        h_freq=high,
                        verbose=False
                    )
                )

                # ------------------------------------------------
                # Filter TEST DATA
                # ------------------------------------------------

                test_band = (
                    mne.filter.filter_data(
                        X_test,
                        sfreq=250,
                        l_freq=low,
                        h_freq=high,
                        verbose=False
                    )
                )

                # ------------------------------------------------
                # CSP for this frequency band
                # ------------------------------------------------

                csp = CSP(
                    n_components=4,
                    reg=None,
                    log=True,
                    norm_trace=False
                )

                train_csp = (
                    csp.fit_transform(
                        train_band,
                        y_train
                    )
                )

                test_csp = (
                    csp.transform(
                        test_band
                    )
                )

                train_features.append(
                    train_csp
                )

                test_features.append(
                    test_csp
                )

            # =================================================
            # CONCATENATE FILTER-BANK FEATURES
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
                "FBCSP training shape:",
                X_train_fbcsp.shape
            )

            print(
                "FBCSP testing shape:",
                X_test_fbcsp.shape
            )

            # ------------------------------------------------
            # Train classifier
            # ------------------------------------------------

            classifier.fit(
                X_train_fbcsp,
                y_train
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            y_pred = (
                classifier.predict(
                    X_test_fbcsp
                )
            )

            accuracy = (
                accuracy_score(
                    y_test,
                    y_pred
                ) * 100
            )

            fold_accuracies.append(
                accuracy
            )

            y_true_all.extend(
                y_test.tolist()
            )

            y_pred_all.extend(
                y_pred.tolist()
            )

            print(
                f"Accuracy: "
                f"{accuracy:.2f}%"
            )

        # ----------------------------------------------------
        # Convert predictions
        # ----------------------------------------------------

        y_true_all = np.asarray(
            y_true_all
        )

        y_pred_all = np.asarray(
            y_pred_all
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=[
                0,
                1
            ]
        )

        normalized_cm = (
            cm.astype(float)
            /
            cm.sum(
                axis=1,
                keepdims=True
            )
        )

        overall_accuracy = (
            accuracy_score(
                y_true_all,
                y_pred_all
            ) * 100
        )

        results[
            classifier_name
        ] = {

            "fold_accuracies":
                np.asarray(
                    fold_accuracies
                ),

            "mean_accuracy":
                np.mean(
                    fold_accuracies
                ),

            "overall_accuracy":
                overall_accuracy,

            "y_true":
                y_true_all,

            "y_pred":
                y_pred_all,

            "confusion_matrix":
                cm,

            "normalized_confusion_matrix":
                normalized_cm
        }

        print(
            f"\nOverall accuracy: "
            f"{overall_accuracy:.2f}%"
        )

        print(
            "\nConfusion matrix:"
        )

        print(cm)

    return results


# ============================================================
# SAVE CONFUSION MATRIX CSV
# ============================================================

def save_confusion_matrix_csv(
    cm,
    output_file
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
            "",
            "Predicted Left",
            "Predicted Right"
        ])

        writer.writerow([
            "Actual Left",
            cm[0, 0],
            cm[0, 1]
        ])

        writer.writerow([
            "Actual Right",
            cm[1, 0],
            cm[1, 1]
        ])


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    cm,
    classifier_name,
    subject,
    method,
    normalized=False
):

    if normalized:

        matrix = (
            cm.astype(float)
            /
            cm.sum(
                axis=1,
                keepdims=True
            )
        )

        values = matrix

        title = (
            f"{method} - "
            f"{classifier_name} - "
            f"{subject}\n"
            f"Normalized Confusion Matrix"
        )

        fmt = ".2f"

    else:

        values = cm

        title = (
            f"{method} - "
            f"{classifier_name} - "
            f"{subject}\n"
            f"Confusion Matrix"
        )

        fmt = "d"

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=values,
        display_labels=[
            "Left",
            "Right"
        ]
    )

    display.plot(
        ax=ax,
        values_format=fmt,
        colorbar=True
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    plt.tight_layout()

    return fig


# ============================================================
# SAVE SUBJECT RESULTS
# ============================================================

def save_subject_confusion_results(
    subject,
    results,
    method,
    output_dir
):

    for classifier_name, data in (
        results.items()
    ):

        classifier_dir = (
            output_dir /
            subject
        )

        classifier_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        safe_name = (
            classifier_name
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_file = (
            classifier_dir /
            f"{safe_name}_confusion_matrix.csv"
        )

        save_confusion_matrix_csv(
            data[
                "confusion_matrix"
            ],
            csv_file
        )

        # ----------------------------------------------------
        # Raw matrix plot
        # ----------------------------------------------------

        fig = plot_confusion_matrix(
            data[
                "confusion_matrix"
            ],
            classifier_name,
            subject,
            method,
            normalized=False
        )

        fig.savefig(
            classifier_dir /
            f"{safe_name}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Normalized plot
        # ----------------------------------------------------

        fig = plot_confusion_matrix(
            data[
                "confusion_matrix"
            ],
            classifier_name,
            subject,
            method,
            normalized=True
        )

        fig.savefig(
            classifier_dir /
            f"{safe_name}_normalized.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


# ============================================================
# AGGREGATE CONFUSION MATRICES
# ============================================================

def calculate_aggregate_confusion_matrix(
    all_results,
    classifier_name
):

    aggregate = np.zeros(
        (2, 2),
        dtype=int
    )

    for subject_results in (
        all_results.values()
    ):

        if classifier_name not in (
            subject_results
        ):
            continue

        aggregate += (
            subject_results[
                classifier_name
            ][
                "confusion_matrix"
            ]
        )

    return aggregate


# ============================================================
# SAVE AGGREGATE MATRICES
# ============================================================

def save_aggregate_results(
    all_results,
    method
):

    print_section(
        f"{method.upper()} AGGREGATE CONFUSION MATRICES"
    )

    output_dir = (
        AGGREGATE_DIR /
        method.lower()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    classifiers = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    for classifier_name in classifiers:

        cm = (
            calculate_aggregate_confusion_matrix(
                all_results,
                classifier_name
            )
        )

        print(
            f"\n{classifier_name}"
        )

        print(cm)

        safe_name = (
            classifier_name
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        save_confusion_matrix_csv(
            cm,
            output_dir /
            f"{safe_name}_aggregate.csv"
        )

        # ----------------------------------------------------
        # Raw plot
        # ----------------------------------------------------

        fig = plot_confusion_matrix(
            cm,
            classifier_name,
            "All Subjects",
            method,
            normalized=False
        )

        fig.savefig(
            output_dir /
            f"{safe_name}_aggregate.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Normalized plot
        # ----------------------------------------------------

        fig = plot_confusion_matrix(
            cm,
            classifier_name,
            "All Subjects",
            method,
            normalized=True
        )

        fig.savefig(
            output_dir /
            f"{safe_name}_aggregate_normalized.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


# ============================================================
# SUBJECT SUMMARY
# ============================================================

def print_subject_summary(
    subject,
    standard_results,
    fbcsp_results
):

    print_section(
        f"{subject} CONFUSION MATRIX SUMMARY"
    )

    classifiers = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    print(
        f"{'Classifier':<25}"
        f"{'Standard CSP':>18}"
        f"{'FBCSP':>18}"
        f"{'Change':>15}"
    )

    print("-" * 80)

    for classifier in classifiers:

        standard_accuracy = (
            standard_results[
                classifier
            ][
                "overall_accuracy"
            ]
        )

        fbcsp_accuracy = (
            fbcsp_results[
                classifier
            ][
                "overall_accuracy"
            ]
        )

        change = (
            fbcsp_accuracy
            -
            standard_accuracy
        )

        print(
            f"{classifier:<25}"
            f"{standard_accuracy:>16.2f}%"
            f"{fbcsp_accuracy:>16.2f}%"
            f"{change:>13.2f} pp"
        )


# ============================================================
# PROCESS SUBJECT
# ============================================================

def process_subject(
    subject,
    file_path
):

    print_section(
        f"PROCESSING {subject}"
    )

    print(
        f"Dataset: {file_path}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    loader = EEGLoader()

    raw = loader.load(
        str(file_path)
    )

    # --------------------------------------------------------
    # Epoch
    # --------------------------------------------------------

    epochs = (
        EEGEpoching
        .create_motor_imagery_epochs(
            raw
        )
    )

    print(
        "Epochs:",
        len(epochs)
    )

    # --------------------------------------------------------
    # Filter 8-30 Hz
    # --------------------------------------------------------

    filtered_epochs = (
        EEGFilter.bandpass(
            epochs
        )
    )

    # --------------------------------------------------------
    # Classification data
    # --------------------------------------------------------

    X, y, class_names = (
        prepare_classification_data(
            filtered_epochs,
            CONDITIONS,
            tmin=CLASSIFICATION_TMIN,
            tmax=CLASSIFICATION_TMAX
        )
    )

    # --------------------------------------------------------
    # Standard CSP
    # --------------------------------------------------------

    standard_results = (
        run_standard_csp_cv(
            X,
            y,
            n_splits=N_SPLITS,
            n_components=N_COMPONENTS
        )
    )

    # --------------------------------------------------------
    # FBCSP
    # --------------------------------------------------------

    fbcsp_results = (
        run_fbcsp_cv(
            X,
            y,
            n_splits=N_SPLITS
        )
    )

    # --------------------------------------------------------
    # Save standard CSP matrices
    # --------------------------------------------------------

    save_subject_confusion_results(
        subject,
        standard_results,
        "Standard CSP",
        STANDARD_DIR
    )

    # --------------------------------------------------------
    # Save FBCSP matrices
    # --------------------------------------------------------

    save_subject_confusion_results(
        subject,
        fbcsp_results,
        "FBCSP",
        FBCSP_DIR
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_subject_summary(
        subject,
        standard_results,
        fbcsp_results
    )

    return (
        standard_results,
        fbcsp_results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("EIDOS-BCI")
    print("CONFUSION MATRIX EXPERIMENT")
    print("=" * 80)

    print(
        "\nDataset: "
        "BCI Competition IV Dataset 2a"
    )

    print(
        "Subjects:",
        ", ".join(SUBJECTS)
    )

    print(
        "Classes:",
        ", ".join(CONDITIONS)
    )

    print(
        "Filter:",
        f"{FILTER_LOW}-{FILTER_HIGH} Hz"
    )

    print(
        "Classification window:",
        f"{CLASSIFICATION_TMIN}-"
        f"{CLASSIFICATION_TMAX} seconds"
    )

    print(
        "CSP components:",
        N_COMPONENTS
    )

    print(
        "Cross-validation:",
        f"{N_SPLITS}-fold StratifiedKFold"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    STANDARD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FBCSP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    AGGREGATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find subjects
    # --------------------------------------------------------

    subject_files = (
        find_subject_files()
    )

    standard_all = {}
    fbcsp_all = {}

    # --------------------------------------------------------
    # Process subjects
    # --------------------------------------------------------

    for subject, file_path in (
        subject_files.items()
    ):

        try:

            (
                standard_results,
                fbcsp_results
            ) = process_subject(
                subject,
                file_path
            )

            standard_all[
                subject
            ] = standard_results

            fbcsp_all[
                subject
            ] = fbcsp_results

            print(
                f"\n✓ {subject} completed."
            )

        except Exception as error:

            print(
                f"\n✗ ERROR processing "
                f"{subject}:"
            )

            print(
                error
            )

            print(
                "\nContinuing..."
            )

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if not standard_all:

        raise RuntimeError(
            "No subjects were successfully processed."
        )

    # --------------------------------------------------------
    # Aggregate matrices
    # --------------------------------------------------------

    save_aggregate_results(
        standard_all,
        "Standard CSP"
    )

    save_aggregate_results(
        fbcsp_all,
        "FBCSP"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CONFUSION MATRIX EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"Subjects successfully processed: "
        f"{len(standard_all)} / "
        f"{len(SUBJECTS)}"
    )

    print(
        "\nStandard CSP results:"
    )

    print(
        STANDARD_DIR
    )

    print(
        "\nFBCSP results:"
    )

    print(
        FBCSP_DIR
    )

    print(
        "\nAggregate results:"
    )

    print(
        AGGREGATE_DIR
    )

    print(
        "\nNext stage:"
    )

    print(
        "Statistical significance testing"
    )

    print("=" * 80)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()