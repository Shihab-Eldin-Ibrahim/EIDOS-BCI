# ============================================================
# EIDOS-BCI
# FOUR-CLASS STATISTICAL SIGNIFICANCE TESTING
#
# Standard CSP vs FBCSP
#
# Dataset:
# BCI Competition IV Dataset 2a
#
# Statistical unit:
# Subject
#
# Tests:
# - Shapiro-Wilk normality test
# - Paired t-test
# - Wilcoxon signed-rank test
# - Cohen's d effect size
#
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.stats import (
    shapiro,
    ttest_rel,
    wilcoxon
)


# ============================================================
# CONFIGURATION
# ============================================================

ALPHA = 0.05

EXPECTED_CLASSIFIERS = [
    "LDA",
    "SVM",
    "Logistic Regression"
]

EXPECTED_SUBJECTS = [
    f"A{i:02d}"
    for i in range(1, 10)
]


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is expected to be:
#
# D:\EIDOS-BCI\analysis\four_class_statistical_tests.py
#
# Therefore:
#
# parent      = analysis
# parent.parent = EIDOS-BCI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)

FOUR_CLASS_DIR = (
    RESULTS_DIR /
    "four_class"
)


STANDARD_CSP_FILE = (
    FOUR_CLASS_DIR /
    "four_class_standard_csp_results.csv"
)

FBCSP_FILE = (
    FOUR_CLASS_DIR /
    "four_class_fbcsp_results.csv"
)

STATISTICS_DIR = (
    RESULTS_DIR /
    "four_class_statistical"
)

STATISTICS_CSV = (
    STATISTICS_DIR /
    "four_class_statistical_comparison.csv"
)

SUBJECT_RESULTS_CSV = (
    STATISTICS_DIR /
    "four_class_subject_improvements.csv"
)

COMPARISON_PLOT = (
    STATISTICS_DIR /
    "four_class_statistical_comparison.png"
)

IMPROVEMENT_PLOT = (
    STATISTICS_DIR /
    "four_class_improvements.png"
)


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
# LOAD RESULTS
# ============================================================

def load_results():

    print_section(
        "LOADING FOUR-CLASS EXPERIMENT RESULTS"
    )

    print(
        "\nDataset:"
    )

    print(
        "BCI Competition IV Dataset 2a"
    )

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        "\nResults directory:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nStandard CSP file:"
    )

    print(
        STANDARD_CSP_FILE
    )

    print(
        "\nFBCSP file:"
    )

    print(
        FBCSP_FILE
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not STANDARD_CSP_FILE.exists():

        raise FileNotFoundError(
            "\nStandard CSP results not found:\n"
            f"{STANDARD_CSP_FILE}"
        )

    if not FBCSP_FILE.exists():

        raise FileNotFoundError(
            "\nFBCSP results not found:\n"
            f"{FBCSP_FILE}"
        )

    print(
        "\n✓ Standard CSP results found."
    )

    print(
        "✓ FBCSP results found."
    )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    baseline = pd.read_csv(
        STANDARD_CSP_FILE
    )

    fbcsp = pd.read_csv(
        FBCSP_FILE
    )

    print(
        "\nStandard CSP data:"
    )

    print(
        baseline.to_string(index=False)
    )

    print(
        "\nFBCSP data:"
    )

    print(
        fbcsp.to_string(index=False)
    )

    return baseline, fbcsp


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(
    baseline,
    fbcsp
):

    print_section(
        "VALIDATING DATA"
    )

    required_columns = {
        "Subject",
        "Experiment",
        "Classifier",
        "Mean Accuracy (%)",
        "Std (%)",
        "Minimum (%)",
        "Maximum (%)"
    }

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    baseline_columns = set(
        baseline.columns
    )

    fbcsp_columns = set(
        fbcsp.columns
    )

    missing_baseline = (
        required_columns
        - baseline_columns
    )

    missing_fbcsp = (
        required_columns
        - fbcsp_columns
    )

    if missing_baseline:

        raise ValueError(
            "Missing columns in Standard CSP file:\n"
            f"{missing_baseline}"
        )

    if missing_fbcsp:

        raise ValueError(
            "Missing columns in FBCSP file:\n"
            f"{missing_fbcsp}"
        )

    print(
        "\n✓ Required columns found."
    )

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    baseline_subjects = sorted(
        baseline["Subject"].unique()
    )

    fbcsp_subjects = sorted(
        fbcsp["Subject"].unique()
    )

    print(
        "\nStandard CSP subjects:"
    )

    print(
        baseline_subjects
    )

    print(
        "\nFBCSP subjects:"
    )

    print(
        fbcsp_subjects
    )

    if baseline_subjects != fbcsp_subjects:

        raise ValueError(
            "Subject sets do not match."
        )

    print(
        "\n✓ Subject sets match."
    )

    # --------------------------------------------------------
    # Classifiers
    # --------------------------------------------------------

    baseline_classifiers = set(
        baseline["Classifier"].unique()
    )

    fbcsp_classifiers = set(
        fbcsp["Classifier"].unique()
    )

    print(
        "\nStandard CSP classifiers:"
    )

    print(
        baseline_classifiers
    )

    print(
        "\nFBCSP classifiers:"
    )

    print(
        fbcsp_classifiers
    )

    if baseline_classifiers != fbcsp_classifiers:

        raise ValueError(
            "Classifier sets do not match."
        )

    print(
        "\n✓ Classifier sets match."
    )

    # --------------------------------------------------------
    # Expected classifiers
    # --------------------------------------------------------

    for classifier in EXPECTED_CLASSIFIERS:

        if classifier not in baseline_classifiers:

            raise ValueError(
                f"Missing classifier: {classifier}"
            )

    print(
        "✓ All expected classifiers found."
    )

    # --------------------------------------------------------
    # Expected subjects
    # --------------------------------------------------------

    for subject in EXPECTED_SUBJECTS:

        if subject not in baseline_subjects:

            raise ValueError(
                f"Missing subject: {subject}"
            )

    print(
        "✓ All expected subjects found."
    )

    # --------------------------------------------------------
    # Check duplicates
    # --------------------------------------------------------

    baseline_duplicates = baseline.duplicated(
        subset=[
            "Subject",
            "Classifier"
        ]
    ).sum()

    fbcsp_duplicates = fbcsp.duplicated(
        subset=[
            "Subject",
            "Classifier"
        ]
    ).sum()

    if baseline_duplicates > 0:

        raise ValueError(
            "Duplicate Subject/Classifier "
            "entries found in Standard CSP."
        )

    if fbcsp_duplicates > 0:

        raise ValueError(
            "Duplicate Subject/Classifier "
            "entries found in FBCSP."
        )

    print(
        "✓ No duplicate subject/classifier entries."
    )


# ============================================================
# PREPARE SUBJECT-LEVEL DATA
# ============================================================

def prepare_subject_results(
    baseline,
    fbcsp,
    classifier
):

    baseline_classifier = (
        baseline[
            baseline["Classifier"] == classifier
        ]
        .copy()
    )

    fbcsp_classifier = (
        fbcsp[
            fbcsp["Classifier"] == classifier
        ]
        .copy()
    )

    baseline_classifier = (
        baseline_classifier[
            [
                "Subject",
                "Mean Accuracy (%)"
            ]
        ]
        .rename(
            columns={
                "Mean Accuracy (%)":
                    "Standard CSP"
            }
        )
    )

    fbcsp_classifier = (
        fbcsp_classifier[
            [
                "Subject",
                "Mean Accuracy (%)"
            ]
        ]
        .rename(
            columns={
                "Mean Accuracy (%)":
                    "FBCSP"
            }
        )
    )

    # --------------------------------------------------------
    # Merge by subject
    # --------------------------------------------------------

    merged = pd.merge(
        baseline_classifier,
        fbcsp_classifier,
        on="Subject",
        how="inner"
    )

    merged = merged.sort_values(
        "Subject"
    )

    # --------------------------------------------------------
    # Calculate improvement
    # --------------------------------------------------------

    merged["Improvement"] = (
        merged["FBCSP"]
        - merged["Standard CSP"]
    )

    return merged


# ============================================================
# COHEN'S D
# ============================================================

def calculate_cohens_d(
    differences
):

    differences = np.asarray(
        differences,
        dtype=float
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
        mean_difference /
        std_difference
    )


# ============================================================
# EFFECT SIZE INTERPRETATION
# ============================================================

def interpret_effect_size(
    d
):

    absolute_d = abs(d)

    if absolute_d < 0.2:

        return "negligible"

    elif absolute_d < 0.5:

        return "small"

    elif absolute_d < 0.8:

        return "medium"

    else:

        return "large"


# ============================================================
# STATISTICAL TEST
# ============================================================

def run_statistical_test(
    subject_results,
    classifier
):

    baseline_values = (
        subject_results[
            "Standard CSP"
        ]
        .to_numpy(
            dtype=float
        )
    )

    fbcsp_values = (
        subject_results[
            "FBCSP"
        ]
        .to_numpy(
            dtype=float
        )
    )

    differences = (
        fbcsp_values
        - baseline_values
    )

    # ========================================================
    # DESCRIPTIVE STATISTICS
    # ========================================================

    baseline_mean = np.mean(
        baseline_values
    )

    baseline_std = np.std(
        baseline_values,
        ddof=1
    )

    fbcsp_mean = np.mean(
        fbcsp_values
    )

    fbcsp_std = np.std(
        fbcsp_values,
        ddof=1
    )

    improvement_mean = np.mean(
        differences
    )

    improvement_std = np.std(
        differences,
        ddof=1
    )

    improvement_min = np.min(
        differences
    )

    improvement_max = np.max(
        differences
    )

    # ========================================================
    # SHAPIRO-WILK
    # ========================================================

    shapiro_statistic, shapiro_p = (
        shapiro(
            differences
        )
    )

    # ========================================================
    # PAIRED T-TEST
    # ========================================================

    t_statistic, t_p = (
        ttest_rel(
            baseline_values,
            fbcsp_values
        )
    )

    # ========================================================
    # WILCOXON
    # ========================================================

    try:

        wilcoxon_statistic, wilcoxon_p = (
            wilcoxon(
                baseline_values,
                fbcsp_values,
                alternative="two-sided"
            )
        )

    except ValueError:

        wilcoxon_statistic = np.nan
        wilcoxon_p = np.nan

    # ========================================================
    # COHEN'S D
    # ========================================================

    cohens_d = calculate_cohens_d(
        differences
    )

    effect_interpretation = (
        interpret_effect_size(
            cohens_d
        )
    )

    # ========================================================
    # NORMALITY INTERPRETATION
    # ========================================================

    if shapiro_p >= ALPHA:

        normality_result = (
            "Differences are consistent "
            "with normality."
        )

        t_test_appropriate = True

    else:

        normality_result = (
            "Differences significantly "
            "deviate from normality."
        )

        t_test_appropriate = False

    # ========================================================
    # SIGNIFICANCE
    # ========================================================

    if t_p < ALPHA:

        t_result = (
            "statistically significant"
        )

    else:

        t_result = (
            "not statistically significant"
        )

    if wilcoxon_p < ALPHA:

        wilcoxon_result = (
            "statistically significant"
        )

    else:

        wilcoxon_result = (
            "not statistically significant"
        )

    # ========================================================
    # PRINT
    # ========================================================

    print_subsection(
        classifier
    )

    print(
        f"\nSubjects: "
        f"{len(subject_results)}"
    )

    print(
        "\nStandard CSP:"
    )

    print(
        f"Mean: {baseline_mean:.2f}%"
    )

    print(
        f"Std: {baseline_std:.2f}%"
    )

    print(
        "\nFBCSP:"
    )

    print(
        f"Mean: {fbcsp_mean:.2f}%"
    )

    print(
        f"Std: {fbcsp_std:.2f}%"
    )

    print(
        "\nMean improvement:"
    )

    print(
        f"{improvement_mean:+.2f} percentage points"
    )

    print(
        f"Improvement Std: "
        f"{improvement_std:.2f}"
    )

    print(
        f"Minimum improvement: "
        f"{improvement_min:+.2f}"
    )

    print(
        f"Maximum improvement: "
        f"{improvement_max:+.2f}"
    )

    # --------------------------------------------------------
    # Normality
    # --------------------------------------------------------

    print(
        "\nNORMALITY TEST"
    )

    print(
        f"Shapiro-Wilk statistic: "
        f"{shapiro_statistic:.4f}"
    )

    print(
        f"Shapiro-Wilk p-value: "
        f"{shapiro_p:.6f}"
    )

    print(
        normality_result
    )

    if t_test_appropriate:

        print(
            "Paired t-test is appropriate."
        )

    else:

        print(
            "Wilcoxon test should be "
            "given greater emphasis."
        )

    # --------------------------------------------------------
    # T-test
    # --------------------------------------------------------

    print(
        "\nPAIRED T-TEST"
    )

    print(
        f"t-statistic: "
        f"{t_statistic:.4f}"
    )

    print(
        f"p-value: "
        f"{t_p:.6f}"
    )

    print(
        f"Result: {t_result}"
    )

    # --------------------------------------------------------
    # Wilcoxon
    # --------------------------------------------------------

    print(
        "\nWILCOXON SIGNED-RANK TEST"
    )

    print(
        f"Wilcoxon statistic: "
        f"{wilcoxon_statistic:.4f}"
    )

    print(
        f"p-value: "
        f"{wilcoxon_p:.6f}"
    )

    print(
        f"Result: {wilcoxon_result}"
    )

    # --------------------------------------------------------
    # Effect size
    # --------------------------------------------------------

    print(
        "\nEFFECT SIZE"
    )

    print(
        f"Cohen's d: "
        f"{cohens_d:.4f}"
    )

    print(
        f"Effect size: "
        f"{effect_interpretation}"
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "Classifier":
            classifier,

        "Subjects":
            len(subject_results),

        "Standard CSP Mean (%)":
            baseline_mean,

        "Standard CSP Std (%)":
            baseline_std,

        "FBCSP Mean (%)":
            fbcsp_mean,

        "FBCSP Std (%)":
            fbcsp_std,

        "Mean Improvement (pp)":
            improvement_mean,

        "Improvement Std (pp)":
            improvement_std,

        "Minimum Improvement (pp)":
            improvement_min,

        "Maximum Improvement (pp)":
            improvement_max,

        "Shapiro Statistic":
            shapiro_statistic,

        "Shapiro p-value":
            shapiro_p,

        "Paired t Statistic":
            t_statistic,

        "Paired t p-value":
            t_p,

        "Wilcoxon Statistic":
            wilcoxon_statistic,

        "Wilcoxon p-value":
            wilcoxon_p,

        "Cohen's d":
            cohens_d,

        "Effect Size":
            effect_interpretation,

        "T-test Significant":
            t_p < ALPHA,

        "Wilcoxon Significant":
            wilcoxon_p < ALPHA
    }


# ============================================================
# PRINT SUBJECT-LEVEL RESULTS
# ============================================================

def print_subject_results(
    subject_results,
    classifier
):

    print_subsection(
        f"{classifier} - SUBJECT-LEVEL IMPROVEMENT"
    )

    for _, row in subject_results.iterrows():

        subject = row["Subject"]

        baseline = row[
            "Standard CSP"
        ]

        fbcsp = row[
            "FBCSP"
        ]

        improvement = row[
            "Improvement"
        ]

        if improvement > 0:

            status = "IMPROVED"

        elif improvement < 0:

            status = "DECREASED"

        else:

            status = "UNCHANGED"

        print(
            f"{subject}: "
            f"{baseline:.2f}% -> "
            f"{fbcsp:.2f}% "
            f"({improvement:+.2f}) "
            f"[{status}]"
        )


# ============================================================
# SAVE SUBJECT RESULTS
# ============================================================

def save_subject_results(
    all_subject_results
):

    STATISTICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for classifier, data in (
        all_subject_results.items()
    ):

        for _, row in data.iterrows():

            rows.append({

                "Subject":
                    row["Subject"],

                "Classifier":
                    classifier,

                "Standard CSP (%)":
                    row["Standard CSP"],

                "FBCSP (%)":
                    row["FBCSP"],

                "Improvement (pp)":
                    row["Improvement"]
            })

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        SUBJECT_RESULTS_CSV,
        index=False
    )

    print(
        "\nSubject-level results saved to:"
    )

    print(
        SUBJECT_RESULTS_CSV
    )


# ============================================================
# SAVE STATISTICAL RESULTS
# ============================================================

def save_statistical_results(
    results
):

    output = pd.DataFrame(
        results
    )

    output.to_csv(
        STATISTICS_CSV,
        index=False
    )

    print(
        "\nStatistical results saved to:"
    )

    print(
        STATISTICS_CSV
    )


# ============================================================
# PLOT MEAN COMPARISON
# ============================================================

def plot_comparison(
    results
):

    classifiers = [
        result["Classifier"]
        for result in results
    ]

    baseline = [
        result[
            "Standard CSP Mean (%)"
        ]
        for result in results
    ]

    fbcsp = [
        result[
            "FBCSP Mean (%)"
        ]
        for result in results
    ]

    x = np.arange(
        len(classifiers)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars1 = ax.bar(
        x - width / 2,
        baseline,
        width,
        label="Standard CSP"
    )

    bars2 = ax.bar(
        x + width / 2,
        fbcsp,
        width,
        label="FBCSP"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    for bar in bars1:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom"
        )

    for bar in bars2:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom"
        )

    ax.set_title(
        "Four-Class Standard CSP vs FBCSP",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Classifier",
        fontsize=14
    )

    ax.set_ylabel(
        "Mean Accuracy Across Subjects (%)",
        fontsize=14
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

    fig.savefig(
        COMPARISON_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "\nComparison plot saved to:"
    )

    print(
        COMPARISON_PLOT
    )


# ============================================================
# PLOT SUBJECT IMPROVEMENTS
# ============================================================

def plot_improvements(
    all_subject_results
):

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    subjects = EXPECTED_SUBJECTS

    x = np.arange(
        len(subjects)
    )

    width = 0.25

    for index, classifier in enumerate(
        EXPECTED_CLASSIFIERS
    ):

        data = (
            all_subject_results[
                classifier
            ]
        )

        data = (
            data
            .set_index("Subject")
            .loc[subjects]
        )

        improvements = (
            data["Improvement"]
            .to_numpy()
        )

        bars = ax.bar(
            x + (
                index - 1
            ) * width,
            improvements,
            width,
            label=classifier
        )

        for bar, value in zip(
            bars,
            improvements
        ):

            vertical_position = (
                value + 0.5
                if value >= 0
                else value - 1.5
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                vertical_position,
                f"{value:+.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # --------------------------------------------------------
    # Zero line
    # --------------------------------------------------------

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_title(
        "Four-Class FBCSP Improvement by Subject",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Subject",
        fontsize=14
    )

    ax.set_ylabel(
        "FBCSP Improvement (Percentage Points)",
        fontsize=14
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        subjects
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        IMPROVEMENT_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "\nImprovement plot saved to:"
    )

    print(
        IMPROVEMENT_PLOT
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    results
):

    print_section(
        "FOUR-CLASS STATISTICAL SUMMARY"
    )

    print(
        f"{'Classifier':<25}"
        f"{'CSP':>12}"
        f"{'FBCSP':>12}"
        f"{'Change':>12}"
        f"{'t p-value':>14}"
        f"{'Wilcoxon p':>16}"
        f"{'Cohen d':>12}"
    )

    print(
        "-" * 105
    )

    for result in results:

        print(
            f"{result['Classifier']:<25}"
            f"{result['Standard CSP Mean (%)']:>10.2f}%"
            f"{result['FBCSP Mean (%)']:>10.2f}%"
            f"{result['Mean Improvement (pp)']:>+10.2f}"
            f"{result['Paired t p-value']:>14.6f}"
            f"{result['Wilcoxon p-value']:>16.6f}"
            f"{result['Cohen\'s d']:>12.4f}"
        )

    print(
        "\n"
    )

    # --------------------------------------------------------
    # Paper interpretation
    # --------------------------------------------------------

    for result in results:

        classifier = result[
            "Classifier"
        ]

        improvement = result[
            "Mean Improvement (pp)"
        ]

        t_p = result[
            "Paired t p-value"
        ]

        wilcoxon_p = result[
            "Wilcoxon p-value"
        ]

        d = result[
            "Cohen's d"
        ]

        effect = result[
            "Effect Size"
        ]

        print(
            f"{classifier}"
        )

        if improvement > 0:

            print(
                f"FBCSP increased mean accuracy "
                f"by {improvement:.2f} percentage points."
            )

        elif improvement < 0:

            print(
                f"FBCSP decreased mean accuracy "
                f"by {abs(improvement):.2f} percentage points."
            )

        else:

            print(
                "FBCSP produced no change "
                "in mean accuracy."
            )

        if (
            t_p < ALPHA
            or
            wilcoxon_p < ALPHA
        ):

            print(
                "Conclusion: The difference is "
                "STATISTICALLY SIGNIFICANT "
                f"(alpha = {ALPHA})."
            )

        else:

            print(
                "Conclusion: The difference is "
                "NOT statistically significant "
                f"(alpha = {ALPHA})."
            )

        print(
            f"Effect size: {effect} "
            f"(Cohen's d = {d:.4f})"
        )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "EIDOS-BCI"
    )

    print(
        "FOUR-CLASS STATISTICAL SIGNIFICANCE TESTING"
    )

    print(
        "STANDARD CSP vs FBCSP"
    )

    print(
        "=" * 100
    )

    print(
        "\nSignificance level:"
    )

    print(
        f"alpha = {ALPHA}"
    )

    print(
        "\nDataset:"
    )

    print(
        "BCI Competition IV Dataset 2a"
    )

    print(
        "\nClassification:"
    )

    print(
        "Four-class motor imagery"
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
        "\nStatistical unit:"
    )

    print(
        "Subject"
    )

    print(
        "\nComparison:"
    )

    print(
        "Standard CSP vs FBCSP"
    )

    print(
        "\nTests:"
    )

    print(
        "Paired t-test"
    )

    print(
        "Wilcoxon signed-rank test"
    )

    print(
        "Shapiro-Wilk normality test"
    )

    print(
        "Cohen's d effect size"
    )

    # ========================================================
    # LOAD
    # ========================================================

    baseline, fbcsp = (
        load_results()
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_data(
        baseline,
        fbcsp
    )

    # ========================================================
    # RUN TESTS
    # ========================================================

    all_subject_results = {}

    statistical_results = []

    for classifier in EXPECTED_CLASSIFIERS:

        subject_results = (
            prepare_subject_results(
                baseline,
                fbcsp,
                classifier
            )
        )

        all_subject_results[
            classifier
        ] = subject_results

        # ----------------------------------------------------
        # Subject-level output
        # ----------------------------------------------------

        print_subject_results(
            subject_results,
            classifier
        )

        # ----------------------------------------------------
        # Statistical test
        # ----------------------------------------------------

        result = (
            run_statistical_test(
                subject_results,
                classifier
            )
        )

        statistical_results.append(
            result
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_subject_results(
        all_subject_results
    )

    save_statistical_results(
        statistical_results
    )

    # ========================================================
    # PLOTS
    # ========================================================

    print_section(
        "GENERATING STATISTICAL PLOTS"
    )

    plot_comparison(
        statistical_results
    )

    plot_improvements(
        all_subject_results
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_final_summary(
        statistical_results
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print_section(
        "FOUR-CLASS STATISTICAL ANALYSIS COMPLETE"
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"\n1. {STATISTICS_CSV}"
    )

    print(
        f"\n2. {SUBJECT_RESULTS_CSV}"
    )

    print(
        f"\n3. {COMPARISON_PLOT}"
    )

    print(
        f"\n4. {IMPROVEMENT_PLOT}"
    )

    print(
        "\nNext research stage:"
    )

    print(
        "1. Subject-independent classification"
    )

    print(
        "2. Cross-subject evaluation"
    )

    print(
        "3. Paper experiment tables"
    )

    print(
        "\n"
    )

    print(
        "=" * 100
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()