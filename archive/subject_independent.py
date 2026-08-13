"""
EIDOS-BCI
SUBJECT-INDEPENDENT STATISTICAL TESTING
STANDARD CSP vs FBCSP

Dataset:
    BCI Competition IV Dataset 2a

Classification:
    Four-class motor imagery

Classes:
    Left Hand
    Right Hand
    Feet
    Tongue

Evaluation:
    Leave-One-Subject-Out (LOSO)

Statistical unit:
    Subject / LOSO fold

Tests:
    - Shapiro-Wilk normality test
    - Paired t-test
    - Wilcoxon signed-rank test
    - Cohen's d effect size

This script compares Standard CSP and FBCSP using the
same LOSO test subjects/folds.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    ttest_rel,
    wilcoxon,
    shapiro,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

ALPHA = 0.05

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "subject_independent"
)

STANDARD_FILE = (
    RESULTS_DIR
    / "subject_independent_standard_csp.csv"
)

FBCSP_FILE = (
    RESULTS_DIR
    / "subject_independent_fbcsp.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "subject_independent_statistical"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUBJECT_IMPROVEMENT_FILE = (
    OUTPUT_DIR
    / "subject_independent_subject_improvements.csv"
)

STATISTICAL_FILE = (
    OUTPUT_DIR
    / "subject_independent_statistical_comparison.csv"
)

COMPARISON_PLOT = (
    OUTPUT_DIR
    / "subject_independent_statistical_comparison.png"
)

IMPROVEMENT_PLOT = (
    OUTPUT_DIR
    / "subject_independent_improvements.png"
)

CLASSIFIERS = [
    "LDA",
    "SVM",
    "Logistic Regression",
]

EXPECTED_SUBJECTS = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A08",
    "A09",
]


# =============================================================================
# PRINT HEADER
# =============================================================================

def print_header():

    print("\n" + "=" * 100)
    print("EIDOS-BCI")
    print("SUBJECT-INDEPENDENT STATISTICAL TESTING")
    print("STANDARD CSP vs FBCSP")
    print("=" * 100)

    print()
    print("Significance level:")
    print(f"alpha = {ALPHA}")

    print()
    print("Dataset:")
    print("BCI Competition IV Dataset 2a")

    print()
    print("Classification:")
    print("Four-class motor imagery")

    print()
    print("Classes:")
    print("Left Hand")
    print("Right Hand")
    print("Feet")
    print("Tongue")

    print()
    print("Evaluation:")
    print("Leave-One-Subject-Out (LOSO)")

    print()
    print("Statistical unit:")
    print("Subject / LOSO fold")

    print()
    print("Comparison:")
    print("Standard CSP vs FBCSP")

    print()
    print("Tests:")
    print("Paired t-test")
    print("Wilcoxon signed-rank test")
    print("Shapiro-Wilk normality test")
    print("Cohen's d effect size")

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Results directory:")
    print(RESULTS_DIR)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"\nResults file not found:\n{path}\n\n"
            "Check that the LOSO experiment was completed."
        )

    df = pd.read_csv(path)

    # Remove accidental whitespace from column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Remove accidental whitespace from string cells
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()

    return df


def load_data():

    print("\n" + "=" * 90)
    print("LOADING LOSO RESULTS")
    print("=" * 90)

    print()
    print("Standard CSP file:")
    print(STANDARD_FILE)

    print()
    print("FBCSP file:")
    print(FBCSP_FILE)

    standard = load_csv(STANDARD_FILE)
    fbcsp = load_csv(FBCSP_FILE)

    print("\n✓ Standard CSP results found.")
    print("✓ FBCSP results found.")

    return standard, fbcsp


# =============================================================================
# NORMALIZE COLUMN NAMES
# =============================================================================

def normalize_subject_column(df, name):

    """
    Accept either:

        Subject

    or:

        Test Subject

    and internally convert it to:

        Subject
    """

    columns = set(df.columns)

    if "Subject" in columns:
        return df

    if "Test Subject" in columns:

        df = df.rename(
            columns={
                "Test Subject": "Subject"
            }
        )

        print(
            f"✓ {name}: converted 'Test Subject' -> 'Subject'"
        )

        return df

    # Additional robust handling
    possible_subject_columns = [
        column
        for column in df.columns
        if str(column).strip().lower()
        in {
            "test subject",
            "test_subject",
            "subject id",
            "subject_id",
            "subject"
        }
    ]

    if len(possible_subject_columns) == 1:

        original = possible_subject_columns[0]

        df = df.rename(
            columns={
                original: "Subject"
            }
        )

        print(
            f"✓ {name}: converted '{original}' -> 'Subject'"
        )

        return df

    raise ValueError(
        f"\nCould not identify subject column in {name}.\n"
        f"Available columns:\n{list(df.columns)}"
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_data(standard, fbcsp):

    print("\n" + "=" * 90)
    print("VALIDATING LOSO DATA")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Normalize subject column
    # -------------------------------------------------------------------------

    standard = normalize_subject_column(
        standard,
        "Standard CSP"
    )

    fbcsp = normalize_subject_column(
        fbcsp,
        "FBCSP"
    )

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    required_columns = {
        "Subject",
        "Experiment",
        "Classifier",
        "Accuracy (%)",
    }

    missing_standard = (
        required_columns
        - set(standard.columns)
    )

    missing_fbcsp = (
        required_columns
        - set(fbcsp.columns)
    )

    if missing_standard:

        raise ValueError(
            "\nMissing columns in Standard CSP:\n"
            f"{missing_standard}\n\n"
            f"Available columns:\n"
            f"{list(standard.columns)}"
        )

    if missing_fbcsp:

        raise ValueError(
            "\nMissing columns in FBCSP:\n"
            f"{missing_fbcsp}\n\n"
            f"Available columns:\n"
            f"{list(fbcsp.columns)}"
        )

    print("\n✓ Required columns found.")

    # -------------------------------------------------------------------------
    # Convert accuracy to numeric
    # -------------------------------------------------------------------------

    standard["Accuracy (%)"] = pd.to_numeric(
        standard["Accuracy (%)"],
        errors="coerce"
    )

    fbcsp["Accuracy (%)"] = pd.to_numeric(
        fbcsp["Accuracy (%)"],
        errors="coerce"
    )

    if standard["Accuracy (%)"].isna().any():

        raise ValueError(
            "Standard CSP contains invalid accuracy values."
        )

    if fbcsp["Accuracy (%)"].isna().any():

        raise ValueError(
            "FBCSP contains invalid accuracy values."
        )

    # -------------------------------------------------------------------------
    # Subject lists
    # -------------------------------------------------------------------------

    standard_subjects = sorted(
        standard["Subject"].unique()
    )

    fbcsp_subjects = sorted(
        fbcsp["Subject"].unique()
    )

    print()
    print("Standard CSP subjects:")
    print(standard_subjects)

    print()
    print("FBCSP subjects:")
    print(fbcsp_subjects)

    if standard_subjects != fbcsp_subjects:

        raise ValueError(
            "\nSubject sets do not match.\n"
            f"Standard CSP: {standard_subjects}\n"
            f"FBCSP: {fbcsp_subjects}"
        )

    print("\n✓ Subject sets match.")

    # -------------------------------------------------------------------------
    # Expected subjects
    # -------------------------------------------------------------------------

    missing_subjects = (
        set(EXPECTED_SUBJECTS)
        - set(standard_subjects)
    )

    if missing_subjects:

        raise ValueError(
            f"\nMissing expected subjects:\n"
            f"{sorted(missing_subjects)}"
        )

    print("✓ All expected subjects found.")

    # -------------------------------------------------------------------------
    # Classifiers
    # -------------------------------------------------------------------------

    standard_classifiers = set(
        standard["Classifier"].unique()
    )

    fbcsp_classifiers = set(
        fbcsp["Classifier"].unique()
    )

    print()
    print("Standard CSP classifiers:")
    print(standard_classifiers)

    print()
    print("FBCSP classifiers:")
    print(fbcsp_classifiers)

    if standard_classifiers != fbcsp_classifiers:

        raise ValueError(
            "\nClassifier sets do not match."
        )

    print("\n✓ Classifier sets match.")

    missing_classifiers = (
        set(CLASSIFIERS)
        - standard_classifiers
    )

    if missing_classifiers:

        raise ValueError(
            f"\nMissing classifiers:\n"
            f"{missing_classifiers}"
        )

    print("✓ All expected classifiers found.")

    # -------------------------------------------------------------------------
    # Duplicate check
    # -------------------------------------------------------------------------

    standard_duplicates = standard.duplicated(
        subset=["Subject", "Classifier"]
    )

    fbcsp_duplicates = fbcsp.duplicated(
        subset=["Subject", "Classifier"]
    )

    if standard_duplicates.any():

        raise ValueError(
            "\nDuplicate Standard CSP subject/classifier entries found."
        )

    if fbcsp_duplicates.any():

        raise ValueError(
            "\nDuplicate FBCSP subject/classifier entries found."
        )

    print("✓ No duplicate subject/classifier entries.")

    return standard, fbcsp


# =============================================================================
# COHEN'S D
# =============================================================================

def cohens_d_paired(
    standard_values,
    fbcsp_values
):

    """
    Cohen's d for paired samples.

    d = mean(differences) / std(differences)

    Difference is defined as:

        FBCSP - Standard CSP
    """

    differences = (
        np.asarray(fbcsp_values)
        - np.asarray(standard_values)
    )

    mean_difference = np.mean(
        differences
    )

    std_difference = np.std(
        differences,
        ddof=1
    )

    if std_difference == 0:

        return np.nan

    return (
        mean_difference
        / std_difference
    )


def interpret_effect_size(d):

    if np.isnan(d):
        return "undefined"

    absolute_d = abs(d)

    if absolute_d < 0.2:
        return "negligible"

    if absolute_d < 0.5:
        return "small"

    if absolute_d < 0.8:
        return "medium"

    return "large"


# =============================================================================
# NORMALITY
# =============================================================================

def normality_test(differences):

    differences = np.asarray(
        differences,
        dtype=float
    )

    if len(differences) < 3:
        return np.nan, np.nan

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        statistic, p_value = shapiro(
            differences
        )

    return statistic, p_value


# =============================================================================
# WILCOXON
# =============================================================================

def wilcoxon_test(
    standard_values,
    fbcsp_values
):

    differences = (
        np.asarray(fbcsp_values)
        - np.asarray(standard_values)
    )

    # If every difference is zero, Wilcoxon cannot
    # produce a meaningful statistic.
    if np.allclose(
        differences,
        0
    ):
        return 0.0, 1.0

    try:

        statistic, p_value = wilcoxon(
            fbcsp_values,
            standard_values,
            alternative="two-sided",
            zero_method="wilcox"
        )

        return statistic, p_value

    except ValueError:

        return np.nan, np.nan


# =============================================================================
# PAIRED ANALYSIS
# =============================================================================

def analyze_classifier(
    standard,
    fbcsp,
    classifier
):

    standard_values = []
    fbcsp_values = []
    subjects = []

    for subject in EXPECTED_SUBJECTS:

        standard_row = standard[
            (standard["Subject"] == subject)
            &
            (
                standard["Classifier"]
                == classifier
            )
        ]

        fbcsp_row = fbcsp[
            (fbcsp["Subject"] == subject)
            &
            (
                fbcsp["Classifier"]
                == classifier
            )
        ]

        if len(standard_row) != 1:

            raise ValueError(
                f"Expected exactly one Standard CSP "
                f"row for {subject}/{classifier}, "
                f"found {len(standard_row)}."
            )

        if len(fbcsp_row) != 1:

            raise ValueError(
                f"Expected exactly one FBCSP "
                f"row for {subject}/{classifier}, "
                f"found {len(fbcsp_row)}."
            )

        standard_accuracy = float(
            standard_row.iloc[0]["Accuracy (%)"]
        )

        fbcsp_accuracy = float(
            fbcsp_row.iloc[0]["Accuracy (%)"]
        )

        subjects.append(subject)

        standard_values.append(
            standard_accuracy
        )

        fbcsp_values.append(
            fbcsp_accuracy
        )

    standard_values = np.array(
        standard_values,
        dtype=float
    )

    fbcsp_values = np.array(
        fbcsp_values,
        dtype=float
    )

    differences = (
        fbcsp_values
        - standard_values
    )

    # -------------------------------------------------------------------------
    # Descriptive statistics
    # -------------------------------------------------------------------------

    standard_mean = np.mean(
        standard_values
    )

    standard_std = np.std(
        standard_values,
        ddof=1
    )

    fbcsp_mean = np.mean(
        fbcsp_values
    )

    fbcsp_std = np.std(
        fbcsp_values,
        ddof=1
    )

    mean_improvement = np.mean(
        differences
    )

    improvement_std = np.std(
        differences,
        ddof=1
    )

    minimum_improvement = np.min(
        differences
    )

    maximum_improvement = np.max(
        differences
    )

    # -------------------------------------------------------------------------
    # Shapiro-Wilk
    # -------------------------------------------------------------------------

    shapiro_statistic, shapiro_p = (
        normality_test(differences)
    )

    # -------------------------------------------------------------------------
    # Paired t-test
    # -------------------------------------------------------------------------

    t_statistic, t_p = ttest_rel(
        fbcsp_values,
        standard_values
    )

    # -------------------------------------------------------------------------
    # Wilcoxon
    # -------------------------------------------------------------------------

    wilcoxon_statistic, wilcoxon_p = (
        wilcoxon_test(
            standard_values,
            fbcsp_values
        )
    )

    # -------------------------------------------------------------------------
    # Cohen's d
    # -------------------------------------------------------------------------

    d = cohens_d_paired(
        standard_values,
        fbcsp_values
    )

    effect_interpretation = (
        interpret_effect_size(d)
    )

    return {
        "Classifier": classifier,

        "N Subjects": len(subjects),

        "Standard CSP Mean (%)":
            standard_mean,

        "Standard CSP Std (%)":
            standard_std,

        "FBCSP Mean (%)":
            fbcsp_mean,

        "FBCSP Std (%)":
            fbcsp_std,

        "Mean Improvement (percentage points)":
            mean_improvement,

        "Improvement Std":
            improvement_std,

        "Minimum Improvement":
            minimum_improvement,

        "Maximum Improvement":
            maximum_improvement,

        "Shapiro-Wilk Statistic":
            shapiro_statistic,

        "Shapiro-Wilk p-value":
            shapiro_p,

        "Paired t-statistic":
            t_statistic,

        "Paired t-test p-value":
            t_p,

        "Wilcoxon Statistic":
            wilcoxon_statistic,

        "Wilcoxon p-value":
            wilcoxon_p,

        "Cohen's d":
            d,

        "Effect Size":
            effect_interpretation,

        "T-test Significant":
            bool(t_p < ALPHA),

        "Wilcoxon Significant":
            bool(wilcoxon_p < ALPHA),
    }


# =============================================================================
# SUBJECT-LEVEL TABLE
# =============================================================================

def create_subject_table(
    standard,
    fbcsp
):

    rows = []

    for classifier in CLASSIFIERS:

        for subject in EXPECTED_SUBJECTS:

            standard_row = standard[
                (standard["Subject"] == subject)
                &
                (
                    standard["Classifier"]
                    == classifier
                )
            ]

            fbcsp_row = fbcsp[
                (fbcsp["Subject"] == subject)
                &
                (
                    fbcsp["Classifier"]
                    == classifier
                )
            ]

            standard_accuracy = float(
                standard_row.iloc[0]["Accuracy (%)"]
            )

            fbcsp_accuracy = float(
                fbcsp_row.iloc[0]["Accuracy (%)"]
            )

            improvement = (
                fbcsp_accuracy
                - standard_accuracy
            )

            if improvement > 0:
                result = "IMPROVED"

            elif improvement < 0:
                result = "DECREASED"

            else:
                result = "UNCHANGED"

            rows.append({
                "Subject": subject,
                "Classifier": classifier,
                "Standard CSP (%)":
                    standard_accuracy,
                "FBCSP (%)":
                    fbcsp_accuracy,
                "Improvement (percentage points)":
                    improvement,
                "Result": result,
            })

    return pd.DataFrame(rows)


# =============================================================================
# PRINT CLASSIFIER ANALYSIS
# =============================================================================

def print_analysis(
    result,
    subject_table
):

    classifier = result["Classifier"]

    print("\n" + "-" * 90)
    print(f"{classifier} - SUBJECT-LEVEL IMPROVEMENT")
    print("-" * 90)

    classifier_data = subject_table[
        subject_table["Classifier"]
        == classifier
    ]

    for _, row in classifier_data.iterrows():

        subject = row["Subject"]

        standard_accuracy = (
            row["Standard CSP (%)"]
        )

        fbcsp_accuracy = (
            row["FBCSP (%)"]
        )

        improvement = (
            row[
                "Improvement (percentage points)"
            ]
        )

        if improvement > 0:
            label = "IMPROVED"

        elif improvement < 0:
            label = "DECREASED"

        else:
            label = "UNCHANGED"

        print(
            f"{subject}: "
            f"{standard_accuracy:.2f}% -> "
            f"{fbcsp_accuracy:.2f}% "
            f"({improvement:+.2f}) "
            f"[{label}]"
        )

    print("\n")
    print(classifier)

    print(
        f"Subjects: "
        f"{result['N Subjects']}"
    )

    print()
    print("Standard CSP:")
    print(
        f"Mean: "
        f"{result['Standard CSP Mean (%)']:.2f}%"
    )
    print(
        f"Std: "
        f"{result['Standard CSP Std (%)']:.2f}%"
    )

    print()
    print("FBCSP:")
    print(
        f"Mean: "
        f"{result['FBCSP Mean (%)']:.2f}%"
    )
    print(
        f"Std: "
        f"{result['FBCSP Std (%)']:.2f}%"
    )

    print()
    print("Mean improvement:")
    print(
        f"{result['Mean Improvement (percentage points)']:+.2f} "
        f"percentage points"
    )

    print(
        f"Improvement Std: "
        f"{result['Improvement Std']:.2f}"
    )

    print(
        f"Minimum improvement: "
        f"{result['Minimum Improvement']:+.2f}"
    )

    print(
        f"Maximum improvement: "
        f"{result['Maximum Improvement']:+.2f}"
    )

    print()
    print("NORMALITY TEST")

    print(
        f"Shapiro-Wilk statistic: "
        f"{result['Shapiro-Wilk Statistic']:.4f}"
    )

    print(
        f"Shapiro-Wilk p-value: "
        f"{result['Shapiro-Wilk p-value']:.6f}"
    )

    if (
        result["Shapiro-Wilk p-value"]
        >= ALPHA
    ):

        print(
            "Differences are consistent "
            "with normality."
        )

        print(
            "Paired t-test is appropriate."
        )

    else:

        print(
            "Differences significantly "
            "deviate from normality."
        )

        print(
            "Wilcoxon test should be "
            "given greater emphasis."
        )

    print()
    print("PAIRED T-TEST")

    print(
        f"t-statistic: "
        f"{result['Paired t-statistic']:.4f}"
    )

    print(
        f"p-value: "
        f"{result['Paired t-test p-value']:.6f}"
    )

    if (
        result["Paired t-test p-value"]
        < ALPHA
    ):

        print(
            "Result: statistically significant"
        )

    else:

        print(
            "Result: not statistically significant"
        )

    print()
    print("WILCOXON SIGNED-RANK TEST")

    print(
        f"Wilcoxon statistic: "
        f"{result['Wilcoxon Statistic']:.4f}"
    )

    print(
        f"p-value: "
        f"{result['Wilcoxon p-value']:.6f}"
    )

    if (
        result["Wilcoxon p-value"]
        < ALPHA
    ):

        print(
            "Result: statistically significant"
        )

    else:

        print(
            "Result: not statistically significant"
        )

    print()
    print("EFFECT SIZE")

    print(
        f"Cohen's d: "
        f"{result['Cohen\'s d']:.4f}"
    )

    print(
        f"Effect size: "
        f"{result['Effect Size']}"
    )


# =============================================================================
# PLOTS
# =============================================================================

def create_comparison_plot(
    results_df
):

    classifiers = results_df[
        "Classifier"
    ].tolist()

    standard_means = results_df[
        "Standard CSP Mean (%)"
    ].values

    fbcsp_means = results_df[
        "FBCSP Mean (%)"
    ].values

    x = np.arange(
        len(classifiers)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        standard_means,
        width,
        label="Standard CSP"
    )

    plt.bar(
        x + width / 2,
        fbcsp_means,
        width,
        label="FBCSP"
    )

    plt.xticks(
        x,
        classifiers
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.xlabel(
        "Classifier"
    )

    plt.title(
        "Subject-Independent LOSO Performance\n"
        "Standard CSP vs FBCSP"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        COMPARISON_PLOT,
        dpi=300
    )

    plt.close()

    print()
    print(
        "Comparison plot saved to:"
    )
    print(COMPARISON_PLOT)


def create_improvement_plot(
    subject_table
):

    classifiers = CLASSIFIERS

    x = np.arange(
        len(EXPECTED_SUBJECTS)
    )

    plt.figure(
        figsize=(12, 7)
    )

    for classifier in classifiers:

        data = subject_table[
            subject_table["Classifier"]
            == classifier
        ]

        improvements = data[
            "Improvement (percentage points)"
        ].values

        plt.plot(
            x,
            improvements,
            marker="o",
            label=classifier
        )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xticks(
        x,
        EXPECTED_SUBJECTS
    )

    plt.xlabel(
        "Test Subject"
    )

    plt.ylabel(
        "FBCSP Improvement (percentage points)"
    )

    plt.title(
        "Subject-Independent LOSO Improvements\n"
        "FBCSP vs Standard CSP"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        IMPROVEMENT_PLOT,
        dpi=300
    )

    plt.close()

    print()
    print(
        "Improvement plot saved to:"
    )
    print(IMPROVEMENT_PLOT)


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(
    results_df
):

    print("\n" + "=" * 100)
    print("SUBJECT-INDEPENDENT STATISTICAL SUMMARY")
    print("=" * 100)

    print()

    print(
        f"{'Classifier':<25}"
        f"{'CSP':>12}"
        f"{'FBCSP':>12}"
        f"{'Change':>12}"
        f"{'t p-value':>15}"
        f"{'Wilcoxon p':>15}"
        f"{'Cohen d':>12}"
    )

    print("-" * 105)

    for _, row in results_df.iterrows():

        print(
            f"{row['Classifier']:<25}"
            f"{row['Standard CSP Mean (%)']:>11.2f}%"
            f"{row['FBCSP Mean (%)']:>11.2f}%"
            f"{row['Mean Improvement (percentage points)']:>+11.2f}"
            f"{row['Paired t-test p-value']:>15.6f}"
            f"{row['Wilcoxon p-value']:>15.6f}"
            f"{row['Cohen\'s d']:>12.4f}"
        )

    print()

    for _, row in results_df.iterrows():

        classifier = row["Classifier"]

        change = row[
            "Mean Improvement (percentage points)"
        ]

        if change > 0:

            direction = (
                f"increased mean accuracy by "
                f"{change:.2f} percentage points"
            )

        elif change < 0:

            direction = (
                f"decreased mean accuracy by "
                f"{abs(change):.2f} percentage points"
            )

        else:

            direction = (
                "produced no mean accuracy change"
            )

        print()
        print(classifier)

        print(
            f"FBCSP {direction}."
        )

        # Primary emphasis:
        # use Wilcoxon when Shapiro rejects normality.
        shapiro_p = row[
            "Shapiro-Wilk p-value"
        ]

        if shapiro_p < ALPHA:

            primary_p = row[
                "Wilcoxon p-value"
            ]

            primary_test = "Wilcoxon"

        else:

            primary_p = row[
                "Paired t-test p-value"
            ]

            primary_test = "paired t-test"

        if primary_p < ALPHA:

            print(
                f"Conclusion: The difference is "
                f"STATISTICALLY SIGNIFICANT "
                f"(alpha = {ALPHA})."
            )

        else:

            print(
                f"Conclusion: The difference is "
                f"NOT statistically significant "
                f"(alpha = {ALPHA})."
            )

        print(
            f"Primary test: {primary_test}"
        )

        print(
            f"Effect size: "
            f"{row['Effect Size']} "
            f"(Cohen's d = "
            f"{row['Cohen\'s d']:.4f})"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header()

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    standard, fbcsp = load_data()

    print()
    print("Standard CSP data:")
    print(standard.to_string(index=False))

    print()
    print("FBCSP data:")
    print(fbcsp.to_string(index=False))

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    standard, fbcsp = validate_data(
        standard,
        fbcsp
    )

    # -------------------------------------------------------------------------
    # Subject-level table
    # -------------------------------------------------------------------------

    subject_table = create_subject_table(
        standard,
        fbcsp
    )

    print("\n" + "=" * 90)
    print("LOSO SUBJECT-LEVEL COMPARISON")
    print("=" * 90)

    print()

    for classifier in CLASSIFIERS:

        print()
        print(classifier)

        data = subject_table[
            subject_table["Classifier"]
            == classifier
        ]

        for _, row in data.iterrows():

            print(
                f"{row['Subject']}: "
                f"{row['Standard CSP (%)']:.2f}% -> "
                f"{row['FBCSP (%)']:.2f}% "
                f"({row['Improvement (percentage points)']:+.2f}) "
                f"[{row['Result']}]"
            )

    # -------------------------------------------------------------------------
    # Statistical analysis
    # -------------------------------------------------------------------------

    results = []

    for classifier in CLASSIFIERS:

        result = analyze_classifier(
            standard,
            fbcsp,
            classifier
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    # -------------------------------------------------------------------------
    # Print detailed analyses
    # -------------------------------------------------------------------------

    for result in results:

        print_analysis(
            result,
            subject_table
        )

    # -------------------------------------------------------------------------
    # Save subject-level results
    # -------------------------------------------------------------------------

    subject_table.to_csv(
        SUBJECT_IMPROVEMENT_FILE,
        index=False
    )

    print()
    print(
        "Subject-level results saved to:"
    )
    print(
        SUBJECT_IMPROVEMENT_FILE
    )

    # -------------------------------------------------------------------------
    # Save statistical results
    # -------------------------------------------------------------------------

    results_df.to_csv(
        STATISTICAL_FILE,
        index=False
    )

    print()
    print(
        "Statistical results saved to:"
    )
    print(
        STATISTICAL_FILE
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    create_comparison_plot(
        results_df
    )

    create_improvement_plot(
        subject_table
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print_summary(
        results_df
    )

    # -------------------------------------------------------------------------
    # Final generated files
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("GENERATED FILES")
    print("=" * 100)

    print()
    print(
        f"1. {STATISTICAL_FILE}"
    )

    print(
        f"2. {SUBJECT_IMPROVEMENT_FILE}"
    )

    print(
        f"3. {COMPARISON_PLOT}"
    )

    print(
        f"4. {IMPROVEMENT_PLOT}"
    )

    print()
    print(
        "Next research stages:"
    )

    print(
        "1. Analyze cross-subject confusion matrices"
    )

    print(
        "2. Compare within-subject vs "
        "subject-independent performance"
    )

    print(
        "3. Generate paper experiment tables"
    )

    print(
        "4. Analyze why FBCSP behaves differently "
        "under cross-subject evaluation"
    )

    print("\n" + "=" * 100)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 100)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()