"""
EIDOS-BCI
CROSS-SUBJECT CONFUSION MATRIX ANALYSIS

Dataset:
    BCI Competition IV Dataset 2a

Classification:
    Four-class Motor Imagery

Classes:
    Left Hand
    Right Hand
    Feet
    Tongue

Evaluation:
    Leave-One-Subject-Out (LOSO)

Methods:
    Standard CSP
    FBCSP

Classifiers:
    LDA
    SVM
    Logistic Regression

Outputs:
    - Per-subject confusion matrix analysis
    - Aggregated confusion matrices
    - Normalized confusion matrices
    - Per-class accuracy
    - Precision
    - Recall
    - F1-score
    - CSP vs FBCSP comparison
    - Most confused class pairs
    - Paper-ready CSV files
    - PNG figures
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = PROJECT_ROOT / "results"

LOSO_ROOT = RESULTS_ROOT / "subject_independent"

CONFUSION_ROOT = LOSO_ROOT / "confusion_matrices"

OUTPUT_ROOT = RESULTS_ROOT / "cross_subject_analysis"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


CLASSES = [
    "Left Hand",
    "Right Hand",
    "Feet",
    "Tongue"
]

CLASSIFIERS = [
    "LDA",
    "SVM",
    "Logistic Regression"
]

METHODS = [
    "Standard CSP",
    "FBCSP"
]


# =============================================================================
# PRINT HEADER
# =============================================================================

def print_header():
    print("=" * 100)
    print("EIDOS-BCI")
    print("CROSS-SUBJECT CONFUSION MATRIX ANALYSIS")
    print("STANDARD CSP vs FBCSP")
    print("=" * 100)

    print()
    print("Dataset:")
    print("BCI Competition IV Dataset 2a")

    print()
    print("Classification:")
    print("Four-class motor imagery")

    print()
    print("Classes:")
    for c in CLASSES:
        print(f"  - {c}")

    print()
    print("Evaluation:")
    print("Leave-One-Subject-Out (LOSO)")

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("LOSO results:")
    print(LOSO_ROOT)

    print()
    print("Confusion matrices:")
    print(CONFUSION_ROOT)

    print()
    print("Output directory:")
    print(OUTPUT_ROOT)

    print()


# =============================================================================
# FIND FILES
# =============================================================================

def find_confusion_files():
    """
    Search recursively for confusion matrix files.

    The exact filenames can vary depending on how the LOSO experiment
    generated them, so this function searches recursively.
    """

    if not CONFUSION_ROOT.exists():
        raise FileNotFoundError(
            f"\nConfusion matrix directory not found:\n"
            f"{CONFUSION_ROOT}\n\n"
            f"Check that the LOSO experiment generated confusion matrices."
        )

    files = list(CONFUSION_ROOT.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"\nNo CSV confusion matrix files found in:\n"
            f"{CONFUSION_ROOT}"
        )

    print("=" * 100)
    print("SEARCHING CONFUSION MATRICES")
    print("=" * 100)
    print()

    print(f"Found {len(files)} CSV files.")

    for file in files:
        print(f"  {file}")

    print()

    return files


# =============================================================================
# METHOD DETECTION
# =============================================================================

def detect_method(path):
    text = str(path).lower()

    if "fbcsp" in text:
        return "FBCSP"

    if "standard" in text or "csp" in text:
        return "Standard CSP"

    return None


# =============================================================================
# CLASSIFIER DETECTION
# =============================================================================

def detect_classifier(path):
    text = str(path).lower()

    if "logistic" in text:
        return "Logistic Regression"

    if "svm" in text:
        return "SVM"

    if "lda" in text:
        return "LDA"

    return None


# =============================================================================
# SUBJECT DETECTION
# =============================================================================

def detect_subject(path):
    """
    Detect subjects such as A01, A02, ..., A09.
    """

    import re

    match = re.search(r"A0?[1-9]", str(path).upper())

    if match:
        return match.group(0).upper()

    return None


# =============================================================================
# READ CONFUSION MATRIX
# =============================================================================

def read_confusion_matrix_file(path):
    """
    Attempt to read a confusion matrix CSV robustly.

    Supports:
        - plain numeric 4x4 matrices
        - matrices with class labels
        - matrices with index columns
    """

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # -------------------------------------------------------------------------
    # Case 1: dataframe contains the four class names
    # -------------------------------------------------------------------------

    normalized_columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    class_names_lower = [
        c.lower()
        for c in CLASSES
    ]

    # -------------------------------------------------------------------------
    # Search for numeric 4x4 matrix
    # -------------------------------------------------------------------------

    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[0] >= 4 and numeric.shape[1] >= 4:

        # Try last 4 numeric columns
        matrix = numeric.iloc[:4, -4:].to_numpy(dtype=float)

        if matrix.shape == (4, 4):
            return matrix

    # -------------------------------------------------------------------------
    # Try raw CSV without header
    # -------------------------------------------------------------------------

    try:
        raw = pd.read_csv(path, header=None)

        numeric_raw = raw.apply(
            pd.to_numeric,
            errors="coerce"
        )

        numeric_raw = numeric_raw.dropna(
            axis=0,
            how="all"
        ).dropna(
            axis=1,
            how="all"
        )

        if numeric_raw.shape[0] >= 4 and numeric_raw.shape[1] >= 4:

            matrix = numeric_raw.iloc[-4:, -4:].to_numpy(
                dtype=float
            )

            if matrix.shape == (4, 4):
                return matrix

    except Exception:
        pass

    return None


# =============================================================================
# LOAD ALL MATRICES
# =============================================================================

def load_matrices(files):

    records = []

    print("=" * 100)
    print("LOADING CONFUSION MATRICES")
    print("=" * 100)
    print()

    for path in files:

        method = detect_method(path)
        classifier = detect_classifier(path)
        subject = detect_subject(path)

        if method is None:
            continue

        if classifier is None:
            continue

        if subject is None:
            continue

        matrix = read_confusion_matrix_file(path)

        if matrix is None:
            print(f"⚠ Could not read matrix:")
            print(f"  {path}")
            continue

        matrix = np.asarray(matrix, dtype=float)

        if matrix.shape != (4, 4):
            print(
                f"⚠ Invalid matrix shape {matrix.shape}: {path}"
            )
            continue

        records.append({
            "Subject": subject,
            "Method": method,
            "Classifier": classifier,
            "Matrix": matrix,
            "File": str(path)
        })

        print(
            f"✓ {subject} | "
            f"{method} | "
            f"{classifier}"
        )

    print()

    if not records:
        raise RuntimeError(
            "No valid confusion matrices could be loaded."
        )

    return records


# =============================================================================
# NORMALIZE MATRIX
# =============================================================================

def normalize_confusion_matrix(matrix):

    matrix = np.asarray(matrix, dtype=float)

    row_sums = matrix.sum(axis=1, keepdims=True)

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0
    )

    return normalized * 100.0


# =============================================================================
# CALCULATE METRICS
# =============================================================================

def calculate_metrics(matrix):

    matrix = np.asarray(matrix, dtype=float)

    total = matrix.sum()

    if total == 0:
        return {
            "Accuracy (%)": np.nan,
            "Macro Precision (%)": np.nan,
            "Macro Recall (%)": np.nan,
            "Macro F1 (%)": np.nan
        }

    accuracy = np.trace(matrix) / total * 100.0

    precision = []
    recall = []
    f1 = []

    for i in range(4):

        tp = matrix[i, i]

        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp

        p = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        r = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        if p + r > 0:
            f = 2 * p * r / (p + r)
        else:
            f = 0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    return {
        "Accuracy (%)": accuracy,
        "Macro Precision (%)": np.mean(precision) * 100,
        "Macro Recall (%)": np.mean(recall) * 100,
        "Macro F1 (%)": np.mean(f1) * 100
    }


# =============================================================================
# PER-CLASS METRICS
# =============================================================================

def calculate_per_class_metrics(matrix):

    matrix = np.asarray(matrix, dtype=float)

    rows = []

    for i, class_name in enumerate(CLASSES):

        tp = matrix[i, i]

        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp

        support = matrix[i, :].sum()

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if precision + recall > 0
            else 0
        )

        rows.append({
            "Class": class_name,
            "Precision (%)": precision * 100,
            "Recall (%)": recall * 100,
            "F1 (%)": f1 * 100,
            "Support": support
        })

    return pd.DataFrame(rows)


# =============================================================================
# SAVE SUBJECT METRICS
# =============================================================================

def create_subject_metrics(records):

    rows = []

    for record in records:

        metrics = calculate_metrics(record["Matrix"])

        row = {
            "Subject": record["Subject"],
            "Method": record["Method"],
            "Classifier": record["Classifier"],
            **metrics
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    output = OUTPUT_ROOT / "subject_metrics.csv"

    df.to_csv(output, index=False)

    print(f"✓ Subject metrics saved:")
    print(f"  {output}")

    return df


# =============================================================================
# AGGREGATE MATRICES
# =============================================================================

def aggregate_matrices(records):

    groups = {}

    for record in records:

        key = (
            record["Method"],
            record["Classifier"]
        )

        if key not in groups:
            groups[key] = np.zeros((4, 4))

        groups[key] += record["Matrix"]

    return groups


# =============================================================================
# SAVE AGGREGATED MATRICES
# =============================================================================

def save_aggregated_matrices(groups):

    rows = []

    for (method, classifier), matrix in groups.items():

        normalized = normalize_confusion_matrix(matrix)

        raw_path = (
            OUTPUT_ROOT /
            f"{method.lower().replace(' ', '_')}_"
            f"{classifier.lower().replace(' ', '_')}_"
            f"confusion_matrix.csv"
        )

        norm_path = (
            OUTPUT_ROOT /
            f"{method.lower().replace(' ', '_')}_"
            f"{classifier.lower().replace(' ', '_')}_"
            f"normalized_confusion_matrix.csv"
        )

        raw_df = pd.DataFrame(
            matrix,
            index=CLASSES,
            columns=CLASSES
        )

        norm_df = pd.DataFrame(
            normalized,
            index=CLASSES,
            columns=CLASSES
        )

        raw_df.to_csv(raw_path)
        norm_df.to_csv(norm_path)

        print(f"✓ {method} | {classifier}")
        print(f"  Raw: {raw_path}")
        print(f"  Normalized: {norm_path}")

        metrics = calculate_metrics(matrix)

        rows.append({
            "Method": method,
            "Classifier": classifier,
            **metrics
        })

    summary = pd.DataFrame(rows)

    summary_path = OUTPUT_ROOT / "aggregated_metrics.csv"

    summary.to_csv(summary_path, index=False)

    print()
    print(f"✓ Aggregated metrics saved:")
    print(f"  {summary_path}")

    return summary


# =============================================================================
# PER-CLASS AGGREGATED METRICS
# =============================================================================

def create_per_class_summary(groups):

    rows = []

    for (method, classifier), matrix in groups.items():

        metrics = calculate_per_class_metrics(matrix)

        for _, row in metrics.iterrows():

            rows.append({
                "Method": method,
                "Classifier": classifier,
                **row.to_dict()
            })

    df = pd.DataFrame(rows)

    path = OUTPUT_ROOT / "per_class_metrics.csv"

    df.to_csv(path, index=False)

    print(f"✓ Per-class metrics saved:")
    print(f"  {path}")

    return df


# =============================================================================
# MOST CONFUSED CLASS PAIRS
# =============================================================================

def find_confused_pairs(groups):

    rows = []

    for (method, classifier), matrix in groups.items():

        normalized = normalize_confusion_matrix(matrix)

        for i in range(4):

            for j in range(4):

                if i == j:
                    continue

                rows.append({
                    "Method": method,
                    "Classifier": classifier,
                    "Actual Class": CLASSES[i],
                    "Predicted Class": CLASSES[j],
                    "Confusion (%)": normalized[i, j]
                })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "Confusion (%)",
        ascending=False
    )

    path = OUTPUT_ROOT / "most_confused_class_pairs.csv"

    df.to_csv(path, index=False)

    print()
    print("=" * 100)
    print("MOST CONFUSED CLASS PAIRS")
    print("=" * 100)
    print()

    for (method, classifier), group in df.groupby(
        ["Method", "Classifier"]
    ):

        print(f"{method} | {classifier}")

        top = group.head(3)

        for _, row in top.iterrows():

            print(
                f"  {row['Actual Class']} -> "
                f"{row['Predicted Class']}: "
                f"{row['Confusion (%)']:.2f}%"
            )

        print()

    print(f"✓ Saved:")
    print(f"  {path}")

    return df


# =============================================================================
# PLOT CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(
    matrix,
    method,
    classifier,
    normalized=True
):

    if normalized:
        data = normalize_confusion_matrix(matrix)
        title_suffix = "Normalized (%)"
        fmt = ".1f"
    else:
        data = matrix
        title_suffix = "Counts"
        fmt = ".0f"

    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(data)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))

    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    ax.set_title(
        f"{method} - {classifier}\n"
        f"Cross-Subject LOSO Confusion Matrix ({title_suffix})"
    )

    for i in range(4):
        for j in range(4):

            ax.text(
                j,
                i,
                format(data[i, j], fmt),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)

    plt.tight_layout()

    filename = (
        f"{method.lower().replace(' ', '_')}_"
        f"{classifier.lower().replace(' ', '_')}_"
        f"{'normalized' if normalized else 'raw'}_confusion.png"
    )

    path = OUTPUT_ROOT / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


# =============================================================================
# GENERATE ALL CONFUSION PLOTS
# =============================================================================

def generate_confusion_plots(groups):

    print()
    print("=" * 100)
    print("GENERATING CONFUSION MATRIX PLOTS")
    print("=" * 100)
    print()

    for (method, classifier), matrix in groups.items():

        path = plot_confusion_matrix(
            matrix,
            method,
            classifier,
            normalized=True
        )

        print(f"✓ {path}")


# =============================================================================
# CSP VS FBCSP COMPARISON
# =============================================================================

def compare_methods(groups):

    rows = []

    for classifier in CLASSIFIERS:

        csp = groups.get(
            ("Standard CSP", classifier)
        )

        fbcsp = groups.get(
            ("FBCSP", classifier)
        )

        if csp is None or fbcsp is None:
            continue

        csp_metrics = calculate_metrics(csp)
        fbcsp_metrics = calculate_metrics(fbcsp)

        rows.append({
            "Classifier": classifier,

            "CSP Accuracy (%)":
                csp_metrics["Accuracy (%)"],

            "FBCSP Accuracy (%)":
                fbcsp_metrics["Accuracy (%)"],

            "Accuracy Change (%)":
                fbcsp_metrics["Accuracy (%)"]
                - csp_metrics["Accuracy (%)"],

            "CSP Macro F1 (%)":
                csp_metrics["Macro F1 (%)"],

            "FBCSP Macro F1 (%)":
                fbcsp_metrics["Macro F1 (%)"],

            "Macro F1 Change (%)":
                fbcsp_metrics["Macro F1 (%)"]
                - csp_metrics["Macro F1 (%)"]
        })

    df = pd.DataFrame(rows)

    path = OUTPUT_ROOT / "csp_vs_fbcsp_comparison.csv"

    df.to_csv(path, index=False)

    print()
    print("=" * 100)
    print("STANDARD CSP vs FBCSP")
    print("=" * 100)
    print()

    print(df.to_string(index=False))

    print()

    print(f"✓ Comparison saved:")
    print(f"  {path}")

    return df


# =============================================================================
# PLOT CSP VS FBCSP
# =============================================================================

def plot_method_comparison(comparison):

    if comparison.empty:
        return

    x = np.arange(len(comparison))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        comparison["CSP Accuracy (%)"],
        width,
        label="Standard CSP"
    )

    ax.bar(
        x + width / 2,
        comparison["FBCSP Accuracy (%)"],
        width,
        label="FBCSP"
    )

    ax.axhline(
        25,
        linestyle="--",
        label="Chance Level (25%)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        comparison["Classifier"]
    )

    ax.set_ylabel("Accuracy (%)")

    ax.set_title(
        "Cross-Subject LOSO Performance\n"
        "Standard CSP vs FBCSP"
    )

    ax.legend()

    ax.set_ylim(
        0,
        max(
            comparison["CSP Accuracy (%)"].max(),
            comparison["FBCSP Accuracy (%)"].max()
        ) + 10
    )

    plt.tight_layout()

    path = (
        OUTPUT_ROOT /
        "csp_vs_fbcsp_cross_subject_accuracy.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Comparison plot saved:")
    print(f"  {path}")


# =============================================================================
# PLOT PER-CLASS PERFORMANCE
# =============================================================================

def plot_per_class_metrics(per_class):

    for classifier in CLASSIFIERS:

        subset = per_class[
            per_class["Classifier"] == classifier
        ]

        if subset.empty:
            continue

        pivot = subset.pivot(
            index="Class",
            columns="Method",
            values="Recall (%)"
        )

        pivot = pivot.reindex(CLASSES)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(CLASSES))
        width = 0.35

        if "Standard CSP" in pivot.columns:

            ax.bar(
                x - width / 2,
                pivot["Standard CSP"],
                width,
                label="Standard CSP"
            )

        if "FBCSP" in pivot.columns:

            ax.bar(
                x + width / 2,
                pivot["FBCSP"],
                width,
                label="FBCSP"
            )

        ax.axhline(
            25,
            linestyle="--",
            label="Chance Level"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            CLASSES,
            rotation=20
        )

        ax.set_ylabel("Recall (%)")

        ax.set_title(
            f"Cross-Subject Per-Class Recall\n"
            f"{classifier}"
        )

        ax.legend()

        plt.tight_layout()

        path = (
            OUTPUT_ROOT /
            f"{classifier.lower().replace(' ', '_')}_"
            f"per_class_recall.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"✓ {path}")


# =============================================================================
# SUBJECT VARIABILITY
# =============================================================================

def analyze_subject_variability(subject_metrics):

    rows = []

    for method in METHODS:

        for classifier in CLASSIFIERS:

            subset = subject_metrics[
                (subject_metrics["Method"] == method) &
                (subject_metrics["Classifier"] == classifier)
            ]

            if subset.empty:
                continue

            values = subset["Accuracy (%)"].values

            rows.append({
                "Method": method,
                "Classifier": classifier,
                "Mean Accuracy (%)": np.mean(values),
                "Std (%)": np.std(values, ddof=1),
                "Minimum (%)": np.min(values),
                "Maximum (%)": np.max(values),
                "Range (%)": np.max(values) - np.min(values)
            })

    df = pd.DataFrame(rows)

    path = OUTPUT_ROOT / "subject_variability.csv"

    df.to_csv(path, index=False)

    print()
    print("=" * 100)
    print("SUBJECT VARIABILITY")
    print("=" * 100)
    print()

    print(df.to_string(index=False))

    print()

    print(f"✓ Subject variability saved:")
    print(f"  {path}")

    return df


# =============================================================================
# FINAL REPORT
# =============================================================================

def print_final_report(
    subject_metrics,
    aggregated,
    comparison
):

    print()
    print("=" * 100)
    print("FINAL CROSS-SUBJECT ANALYSIS")
    print("=" * 100)
    print()

    print("Subjects analyzed:")

    subjects = sorted(
        subject_metrics["Subject"].unique()
    )

    print(
        f"{len(subjects)} / 9"
    )

    print()

    print("Subjects:")

    print(
        ", ".join(subjects)
    )

    print()

    print("AGGREGATED PERFORMANCE")
    print("-" * 100)

    for _, row in aggregated.iterrows():

        print(
            f"{row['Method']} | "
            f"{row['Classifier']}"
        )

        print(
            f"  Accuracy: "
            f"{row['Accuracy (%)']:.2f}%"
        )

        print(
            f"  Macro Precision: "
            f"{row['Macro Precision (%)']:.2f}%"
        )

        print(
            f"  Macro Recall: "
            f"{row['Macro Recall (%)']:.2f}%"
        )

        print(
            f"  Macro F1: "
            f"{row['Macro F1 (%)']:.2f}%"
        )

        print()

    print("METHOD COMPARISON")
    print("-" * 100)

    for _, row in comparison.iterrows():

        change = row["Accuracy Change (%)"]

        direction = (
            "IMPROVED"
            if change > 0
            else "DECREASED"
            if change < 0
            else "UNCHANGED"
        )

        print(
            f"{row['Classifier']}: "
            f"{row['CSP Accuracy (%)']:.2f}% -> "
            f"{row['FBCSP Accuracy (%)']:.2f}% "
            f"({change:+.2f}) "
            f"[{direction}]"
        )

    print()

    print("IMPORTANT:")
    print(
        "These are descriptive cross-subject results."
    )

    print(
        "Statistical significance should be interpreted "
        "using the LOSO statistical testing results."
    )

    print()

    print("Output directory:")
    print(OUTPUT_ROOT)

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header()

    files = find_confusion_files()

    records = load_matrices(files)

    subject_metrics = create_subject_metrics(
        records
    )

    groups = aggregate_matrices(
        records
    )

    aggregated = save_aggregated_matrices(
        groups
    )

    per_class = create_per_class_summary(
        groups
    )

    find_confused_pairs(
        groups
    )

    generate_confusion_plots(
        groups
    )

    comparison = compare_methods(
        groups
    )

    plot_method_comparison(
        comparison
    )

    plot_per_class_metrics(
        per_class
    )

    analyze_subject_variability(
        subject_metrics
    )

    print_final_report(
        subject_metrics,
        aggregated,
        comparison
    )

    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print()

    print("Generated files:")
    print()

    for file in sorted(OUTPUT_ROOT.iterdir()):

        if file.is_file():
            print(f"  {file}")

    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()