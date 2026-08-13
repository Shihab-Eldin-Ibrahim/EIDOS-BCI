from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    shapiro,
    ttest_rel,
    wilcoxon
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "four_class"
    / "cross_subject"
)

STANDARD_CSP_FILE = (
    RESULTS_DIR
    / "standard_csp_loso_results.csv"
)

FBCSP_FILE = (
    RESULTS_DIR
    / "fbcsp_loso_results.csv"
)

STATISTICS_DIR = (
    RESULTS_DIR
    / "statistical_tests"
)

STATISTICS_FILE = (
    STATISTICS_DIR
    / "loso_statistical_comparison.csv"
)

SUBJECT_IMPROVEMENTS_FILE = (
    STATISTICS_DIR
    / "loso_subject_improvements.csv"
)

COMPARISON_PLOT = (
    STATISTICS_DIR
    / "loso_statistical_comparison.png"
)

IMPROVEMENT_PLOT = (
    STATISTICS_DIR
    / "loso_improvements.png"
)

ALPHA = 0.05

CLASSIFIERS = [
    "LDA",
    "SVM",
    "Logistic Regression"
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
# LOAD RESULTS
# ============================================================

def load_results():

    print_section(
        "LOSO STATISTICAL SIGNIFICANCE TESTING"
    )

    print(
        "Dataset:"
        "\nBCI Competition IV Dataset 2a"
    )

    print(
        "\nStatistical unit:"
        "\nSubject"
    )

    print(
        "\nEvaluation:"
        "\nLeave-One-Subject-Out (LOSO)"
    )

    print(
        "\nComparison:"
        "\nStandard CSP vs FBCSP"
    )

    print(
        "\nSignificance level:"
        f"\nalpha = {ALPHA}"
    )

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "\nResults directory:",
        RESULTS_DIR
    )

    print(
        "\nStandard CSP file:",
        STANDARD_CSP_FILE
    )

    print(
        "\nFBCSP file:",
        FBCSP_FILE
    )

    if not STANDARD_CSP_FILE.exists():

        raise FileNotFoundError(
            f"\nStandard CSP file not found:\n"
            f"{STANDARD_CSP_FILE}"
        )

    if not FBCSP_FILE.exists():

        raise FileNotFoundError(
            f"\nFBCSP file not found:\n"
            f"{FBCSP_FILE}"
        )

    print(
        "\n✓ Standard CSP results found."
    )

    print(
        "✓ FBCSP results found."
    )

    standard = pd.read_csv(
        STANDARD_CSP_FILE
    )

    fbcsp = pd.read_csv(
        FBCSP_FILE
    )

    required_columns = [
        "Test Subject",
        "Classifier",
        "Accuracy (%)"
    ]

    for column in required_columns:

        if column not in standard.columns:

            raise ValueError(
                f"Missing column in Standard CSP file: "
                f"{column}"
            )

        if column not in fbcsp.columns:

            raise ValueError(
                f"Missing column in FBCSP file: "
                f"{column}"
            )

    print(
        "\n✓ Required columns found."
    )

    return standard, fbcsp


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(
    standard,
    fbcsp
):

    print_section(
        "VALIDATING LOSO DATA"
    )

    standard_subjects = sorted(
        standard[
            "Test Subject"
        ].unique()
    )

    fbcsp_subjects = sorted(
        fbcsp[
            "Test Subject"
        ].unique()
    )

    print(
        "\nStandard CSP subjects:"
    )

    print(
        standard_subjects
    )

    print(
        "\nFBCSP subjects:"
    )

    print(
        fbcsp_subjects
    )

    if standard_subjects != fbcsp_subjects:

        raise ValueError(
            "Subject sets do not match."
        )

    print(
        "\n✓ Subject sets match."
    )

    standard_classifiers = set(
        standard[
            "Classifier"
        ].unique()
    )

    fbcsp_classifiers = set(
        fbcsp[
            "Classifier"
        ].unique()
    )

    print(
        "\nStandard CSP classifiers:"
    )

    print(
        standard_classifiers
    )

    print(
        "\nFBCSP classifiers:"
    )

    print(
        fbcsp_classifiers
    )

    if (
        standard_classifiers
        != fbcsp_classifiers
    ):

        raise ValueError(
            "Classifier sets do not match."
        )

    if (
        standard_classifiers
        != set(CLASSIFIERS)
    ):

        raise ValueError(
            "Unexpected classifier set."
        )

    print(
        "\n✓ Classifier sets match."
    )

    # --------------------------------------------------------
    # Check duplicates
    # --------------------------------------------------------

    if standard.duplicated(
        subset=[
            "Test Subject",
            "Classifier"
        ]
    ).any():

        raise ValueError(
            "Duplicate Standard CSP "
            "subject/classifier entries."
        )

    if fbcsp.duplicated(
        subset=[
            "Test Subject",
            "Classifier"
        ]
    ).any():

        raise ValueError(
            "Duplicate FBCSP "
            "subject/classifier entries."
        )

    print(
        "✓ No duplicate subject/classifier entries."
    )


# ============================================================
# COHEN'S D
# ============================================================

def cohens_d_paired(
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
        mean_difference
        / std_difference
    )


def interpret_effect_size(
    d
):

    abs_d = abs(d)

    if abs_d < 0.2:

        return "negligible"

    elif abs_d < 0.5:

        return "small"

    elif abs_d < 0.8:

        return "medium"

    else:

        return "large"


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def analyze_classifier(
    standard,
    fbcsp,
    classifier
):

    print_section(
        f"{classifier} - LOSO STATISTICAL ANALYSIS"
    )

    standard_values = []
    fbcsp_values = []
    subjects = []

    for subject in sorted(
        standard[
            "Test Subject"
        ].unique()
    ):

        standard_row = standard[
            (
                standard[
                    "Test Subject"
                ]
                == subject
            )
            &
            (
                standard[
                    "Classifier"
                ]
                == classifier
            )
        ]

        fbcsp_row = fbcsp[
            (
                fbcsp[
                    "Test Subject"
                ]
                == subject
            )
            &
            (
                fbcsp[
                    "Classifier"
                ]
                == classifier
            )
        ]

        if len(standard_row) != 1:

            raise ValueError(
                f"Expected exactly one "
                f"Standard CSP result for "
                f"{subject}, {classifier}"
            )

        if len(fbcsp_row) != 1:

            raise ValueError(
                f"Expected exactly one "
                f"FBCSP result for "
                f"{subject}, {classifier}"
            )

        standard_accuracy = float(
            standard_row[
                "Accuracy (%)"
            ].iloc[0]
        )

        fbcsp_accuracy = float(
            fbcsp_row[
                "Accuracy (%)"
            ].iloc[0]
        )

        subjects.append(
            subject
        )

        standard_values.append(
            standard_accuracy
        )

        fbcsp_values.append(
            fbcsp_accuracy
        )

    standard_values = np.asarray(
        standard_values,
        dtype=float
    )

    fbcsp_values = np.asarray(
        fbcsp_values,
        dtype=float
    )

    differences = (
        fbcsp_values
        - standard_values
    )

    # --------------------------------------------------------
    # Print subject-level improvements
    # --------------------------------------------------------

    print_subsection(
        "SUBJECT-LEVEL IMPROVEMENT"
    )

    for (
        subject,
        baseline,
        fbcsp_accuracy,
        difference
    ) in zip(
        subjects,
        standard_values,
        fbcsp_values,
        differences
    ):

        status = (
            "IMPROVED"
            if difference > 0
            else
            "DECREASED"
            if difference < 0
            else
            "UNCHANGED"
        )

        print(
            f"{subject}: "
            f"{baseline:.2f}% -> "
            f"{fbcsp_accuracy:.2f}% "
            f"({difference:+.2f}) "
            f"[{status}]"
        )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

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

    mean_difference = np.mean(
        differences
    )

    difference_std = np.std(
        differences,
        ddof=1
    )

    minimum_difference = np.min(
        differences
    )

    maximum_difference = np.max(
        differences
    )

    print_subsection(
        "DESCRIPTIVE STATISTICS"
    )

    print(
        f"Subjects: {len(subjects)}"
    )

    print(
        f"\nStandard CSP:"
        f"\nMean: {standard_mean:.2f}%"
        f"\nStd: {standard_std:.2f}%"
    )

    print(
        f"\nFBCSP:"
        f"\nMean: {fbcsp_mean:.2f}%"
        f"\nStd: {fbcsp_std:.2f}%"
    )

    print(
        f"\nMean improvement:"
        f"\n{mean_difference:+.2f} percentage points"
    )

    print(
        f"Improvement Std: "
        f"{difference_std:.2f}"
    )

    print(
        f"Minimum improvement: "
        f"{minimum_difference:+.2f}"
    )

    print(
        f"Maximum improvement: "
        f"{maximum_difference:+.2f}"
    )

    # --------------------------------------------------------
    # Shapiro-Wilk
    # --------------------------------------------------------

    print_subsection(
        "SHAPIRO-WILK NORMALITY TEST"
    )

    shapiro_stat, shapiro_p = (
        shapiro(
            differences
        )
    )

    print(
        f"Shapiro-Wilk statistic: "
        f"{shapiro_stat:.4f}"
    )

    print(
        f"Shapiro-Wilk p-value: "
        f"{shapiro_p:.6f}"
    )

    if shapiro_p > ALPHA:

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

    # --------------------------------------------------------
    # Paired t-test
    # --------------------------------------------------------

    print_subsection(
        "PAIRED T-TEST"
    )

    t_stat, t_p = ttest_rel(
        standard_values,
        fbcsp_values
    )

    print(
        f"t-statistic: "
        f"{t_stat:.4f}"
    )

    print(
        f"p-value: "
        f"{t_p:.6f}"
    )

    if t_p < ALPHA:

        print(
            "Result: statistically significant"
        )

    else:

        print(
            "Result: NOT statistically significant"
        )

    # --------------------------------------------------------
    # Wilcoxon
    # --------------------------------------------------------

    print_subsection(
        "WILCOXON SIGNED-RANK TEST"
    )

    try:

        wilcoxon_stat, wilcoxon_p = (
            wilcoxon(
                standard_values,
                fbcsp_values,
                zero_method="wilcox",
                alternative="two-sided"
            )
        )

        print(
            f"Wilcoxon statistic: "
            f"{wilcoxon_stat:.4f}"
        )

        print(
            f"p-value: "
            f"{wilcoxon_p:.6f}"
        )

        if wilcoxon_p < ALPHA:

            print(
                "Result: statistically significant"
            )

        else:

            print(
                "Result: NOT statistically significant"
            )

    except ValueError:

        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan

        print(
            "Wilcoxon test could not be computed."
        )

    # --------------------------------------------------------
    # Cohen's d
    # --------------------------------------------------------

    print_subsection(
        "EFFECT SIZE"
    )

    d = cohens_d_paired(
        differences
    )

    effect_interpretation = (
        interpret_effect_size(d)
    )

    print(
        f"Cohen's d: "
        f"{d:.4f}"
    )

    print(
        f"Effect size: "
        f"{effect_interpretation}"
    )

    # --------------------------------------------------------
    # Number improved/decreased
    # --------------------------------------------------------

    improved = np.sum(
        differences > 0
    )

    decreased = np.sum(
        differences < 0
    )

    unchanged = np.sum(
        differences == 0
    )

    print_subsection(
        "SUBJECT RESPONSE"
    )

    print(
        f"Improved: {improved}/{len(subjects)}"
    )

    print(
        f"Decreased: {decreased}/{len(subjects)}"
    )

    print(
        f"Unchanged: {unchanged}/{len(subjects)}"
    )

    return {

        "subjects": subjects,

        "standard_values":
            standard_values,

        "fbcsp_values":
            fbcsp_values,

        "differences":
            differences,

        "standard_mean":
            standard_mean,

        "standard_std":
            standard_std,

        "fbcsp_mean":
            fbcsp_mean,

        "fbcsp_std":
            fbcsp_std,

        "mean_difference":
            mean_difference,

        "difference_std":
            difference_std,

        "minimum_difference":
            minimum_difference,

        "maximum_difference":
            maximum_difference,

        "shapiro_stat":
            shapiro_stat,

        "shapiro_p":
            shapiro_p,

        "t_stat":
            t_stat,

        "t_p":
            t_p,

        "wilcoxon_stat":
            wilcoxon_stat,

        "wilcoxon_p":
            wilcoxon_p,

        "cohens_d":
            d,

        "effect":
            effect_interpretation,

        "improved":
            int(improved),

        "decreased":
            int(decreased),

        "unchanged":
            int(unchanged)
    }


# ============================================================
# SAVE STATISTICAL RESULTS
# ============================================================

def save_statistical_results(
    results
):

    STATISTICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        STATISTICS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Classifier",
            "Subjects",
            "Standard CSP Mean (%)",
            "Standard CSP Std (%)",
            "FBCSP Mean (%)",
            "FBCSP Std (%)",
            "Mean Change (pp)",
            "Change Std",
            "Minimum Change (pp)",
            "Maximum Change (pp)",
            "Shapiro Statistic",
            "Shapiro p-value",
            "Paired t-statistic",
            "Paired t p-value",
            "Wilcoxon Statistic",
            "Wilcoxon p-value",
            "Cohen's d",
            "Effect Size",
            "Improved Subjects",
            "Decreased Subjects"
        ])

        for classifier in CLASSIFIERS:

            data = results[
                classifier
            ]

            writer.writerow([
                classifier,
                len(data["subjects"]),
                f"{data['standard_mean']:.4f}",
                f"{data['standard_std']:.4f}",
                f"{data['fbcsp_mean']:.4f}",
                f"{data['fbcsp_std']:.4f}",
                f"{data['mean_difference']:.4f}",
                f"{data['difference_std']:.4f}",
                f"{data['minimum_difference']:.4f}",
                f"{data['maximum_difference']:.4f}",
                f"{data['shapiro_stat']:.6f}",
                f"{data['shapiro_p']:.6f}",
                f"{data['t_stat']:.6f}",
                f"{data['t_p']:.6f}",
                f"{data['wilcoxon_stat']:.6f}",
                f"{data['wilcoxon_p']:.6f}",
                f"{data['cohens_d']:.6f}",
                data["effect"],
                data["improved"],
                data["decreased"]
            ])

    print(
        "\nStatistical results saved to:"
        f"\n{STATISTICS_FILE}"
    )


# ============================================================
# SAVE SUBJECT IMPROVEMENTS
# ============================================================

def save_subject_improvements(
    results
):

    with open(
        SUBJECT_IMPROVEMENTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Test Subject",
            "Classifier",
            "Standard CSP (%)",
            "FBCSP (%)",
            "Improvement (pp)"
        ])

        for classifier in CLASSIFIERS:

            data = results[
                classifier
            ]

            for (
                subject,
                standard,
                fbcsp,
                difference
            ) in zip(
                data["subjects"],
                data["standard_values"],
                data["fbcsp_values"],
                data["differences"]
            ):

                writer.writerow([
                    subject,
                    classifier,
                    f"{standard:.4f}",
                    f"{fbcsp:.4f}",
                    f"{difference:.4f}"
                ])

    print(
        "\nSubject-level improvements saved to:"
        f"\n{SUBJECT_IMPROVEMENTS_FILE}"
    )


# ============================================================
# PLOT COMPARISON
# ============================================================

def plot_comparison(
    results
):

    standard_means = [
        results[
            classifier
        ]["standard_mean"]
        for classifier in CLASSIFIERS
    ]

    fbcsp_means = [
        results[
            classifier
        ]["fbcsp_mean"]
        for classifier in CLASSIFIERS
    ]

    x = np.arange(
        len(CLASSIFIERS)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars1 = ax.bar(
        x - width / 2,
        standard_means,
        width,
        label="Standard CSP"
    )

    bars2 = ax.bar(
        x + width / 2,
        fbcsp_means,
        width,
        label="FBCSP"
    )

    for bar, value in zip(
        bars1,
        standard_means
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
        fbcsp_means
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center"
        )

    ax.axhline(
        25,
        linestyle="--",
        linewidth=1,
        label="Chance level (25%)"
    )

    ax.set_title(
        "LOSO Four-Class Motor Imagery:\n"
        "Standard CSP vs FBCSP"
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
        CLASSIFIERS
    )

    ax.set_ylim(
        0,
        70
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

    print(
        "\nComparison plot saved to:"
        f"\n{COMPARISON_PLOT}"
    )


# ============================================================
# PLOT IMPROVEMENTS
# ============================================================

def plot_improvements(
    results
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for classifier in CLASSIFIERS:

        differences = results[
            classifier
        ]["differences"]

        subjects = results[
            classifier
        ]["subjects"]

        ax.plot(
            subjects,
            differences,
            marker="o",
            label=classifier
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "LOSO Subject-Level FBCSP Improvement"
    )

    ax.set_xlabel(
        "Test Subject"
    )

    ax.set_ylabel(
        "FBCSP - Standard CSP (percentage points)"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        IMPROVEMENT_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nImprovement plot saved to:"
        f"\n{IMPROVEMENT_PLOT}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    results
):

    print_section(
        "LOSO STATISTICAL FINAL RESULTS"
    )

    print(
        f"{'Classifier':<25}"
        f"{'CSP':>12}"
        f"{'FBCSP':>12}"
        f"{'Change':>12}"
        f"{'t p':>14}"
        f"{'Wilcoxon p':>16}"
        f"{'Cohen d':>12}"
    )

    print(
        "-" * 105
    )

    for classifier in CLASSIFIERS:

        data = results[
            classifier
        ]

        print(
            f"{classifier:<25}"
            f"{data['standard_mean']:>10.2f}%"
            f"{data['fbcsp_mean']:>10.2f}%"
            f"{data['mean_difference']:>+10.2f}"
            f"{data['t_p']:>14.6f}"
            f"{data['wilcoxon_p']:>16.6f}"
            f"{data['cohens_d']:>12.4f}"
        )

    print(
        "-" * 105
    )

    for classifier in CLASSIFIERS:

        data = results[
            classifier
        ]

        print(
            f"\n{classifier}"
        )

        if data["wilcoxon_p"] < ALPHA:

            print(
                "Wilcoxon result: "
                "STATISTICALLY SIGNIFICANT"
            )

        else:

            print(
                "Wilcoxon result: "
                "NOT statistically significant"
            )

        if data["t_p"] < ALPHA:

            print(
                "Paired t-test: "
                "STATISTICALLY SIGNIFICANT"
            )

        else:

            print(
                "Paired t-test: "
                "NOT statistically significant"
            )

        print(
            f"Mean change: "
            f"{data['mean_difference']:+.2f} pp"
        )

        print(
            f"Subjects improved: "
            f"{data['improved']}/9"
        )

        print(
            f"Subjects decreased: "
            f"{data['decreased']}/9"
        )

        print(
            f"Cohen's d: "
            f"{data['cohens_d']:.4f} "
            f"({data['effect']})"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    standard, fbcsp = load_results()

    validate_data(
        standard,
        fbcsp
    )

    results = {}

    for classifier in CLASSIFIERS:

        results[
            classifier
        ] = analyze_classifier(
            standard,
            fbcsp,
            classifier
        )

    save_statistical_results(
        results
    )

    save_subject_improvements(
        results
    )

    plot_comparison(
        results
    )

    plot_improvements(
        results
    )

    print_final_summary(
        results
    )

    print_section(
        "LOSO STATISTICAL TESTING COMPLETE"
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {STATISTICS_FILE}"
    )

    print(
        f"2. {SUBJECT_IMPROVEMENTS_FILE}"
    )

    print(
        f"3. {COMPARISON_PLOT}"
    )

    print(
        f"4. {IMPROVEMENT_PLOT}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()