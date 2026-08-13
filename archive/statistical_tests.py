# ============================================================
# EIDOS-BCI
# STATISTICAL COMPARISON
# STANDARD CSP vs FBCSP
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.stats import (
    ttest_rel,
    wilcoxon,
    shapiro
)

# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

# This file is:
#
# D:\EIDOS-BCI\analysis\statistical_tests.py
#
# Therefore:
#
# parent       -> D:\EIDOS-BCI\analysis
# parent.parent -> D:\EIDOS-BCI
#
# This prevents the previous problem where Python searched for:
#
# D:\EIDOS-BCI\analysis\results
#
# instead of:
#
# D:\EIDOS-BCI\results
# ------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)

# ------------------------------------------------------------
# RESULT FILES
# ------------------------------------------------------------

BASELINE_RESULTS_FILE = (
    RESULTS_DIR /
    "multi_subject_baseline_results.csv"
)

FBCSP_RESULTS_FILE = (
    RESULTS_DIR /
    "multi_subject_fbcsp_results.csv"
)

# ------------------------------------------------------------
# Statistical significance level
# ------------------------------------------------------------

ALPHA = 0.05

# ------------------------------------------------------------
# Classifiers
# ------------------------------------------------------------

CLASSIFIERS = [
    "LDA",
    "SVM",
    "Logistic Regression"
]

# ============================================================
# HELPER FUNCTIONS
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
        "LOADING EXPERIMENT RESULTS"
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
        "\nBaseline file:"
    )

    print(
        BASELINE_RESULTS_FILE
    )

    print(
        "\nFBCSP file:"
    )

    print(
        FBCSP_RESULTS_FILE
    )

    # --------------------------------------------------------
    # Check results directory
    # --------------------------------------------------------

    if not RESULTS_DIR.exists():

        raise FileNotFoundError(
            f"\nResults directory not found:\n"
            f"{RESULTS_DIR}"
        )

    # --------------------------------------------------------
    # Check baseline file
    # --------------------------------------------------------

    if not BASELINE_RESULTS_FILE.exists():

        raise FileNotFoundError(
            "\nBaseline results not found:\n"
            f"{BASELINE_RESULTS_FILE}"
        )

    # --------------------------------------------------------
    # Check FBCSP file
    # --------------------------------------------------------

    if not FBCSP_RESULTS_FILE.exists():

        raise FileNotFoundError(
            "\nFBCSP results not found:\n"
            f"{FBCSP_RESULTS_FILE}"
        )

    print(
        "\n✓ Baseline results found."
    )

    print(
        "✓ FBCSP results found."
    )

    # --------------------------------------------------------
    # Load CSV files
    # --------------------------------------------------------

    baseline = pd.read_csv(
        BASELINE_RESULTS_FILE
    )

    fbcsp = pd.read_csv(
        FBCSP_RESULTS_FILE
    )

    print(
        "\nBaseline data:"
    )

    print(
        baseline
    )

    print(
        "\nFBCSP data:"
    )

    print(
        fbcsp
    )

    return (
        baseline,
        fbcsp
    )


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_results(
    baseline,
    fbcsp
):

    print_section(
        "VALIDATING EXPERIMENT RESULTS"
    )

    required_columns = [
        "Subject",
        "Classifier",
        "Mean Accuracy (%)"
    ]

    # --------------------------------------------------------
    # Baseline columns
    # --------------------------------------------------------

    for column in required_columns:

        if column not in baseline.columns:

            raise ValueError(
                f"Baseline CSV is missing "
                f"column: {column}"
            )

    # --------------------------------------------------------
    # FBCSP columns
    # --------------------------------------------------------

    for column in required_columns:

        if column not in fbcsp.columns:

            raise ValueError(
                f"FBCSP CSV is missing "
                f"column: {column}"
            )

    print(
        "✓ Required columns found."
    )

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    baseline_subjects = set(
        baseline["Subject"]
    )

    fbcsp_subjects = set(
        fbcsp["Subject"]
    )

    print(
        "\nBaseline subjects:"
    )

    print(
        sorted(baseline_subjects)
    )

    print(
        "\nFBCSP subjects:"
    )

    print(
        sorted(fbcsp_subjects)
    )

    if baseline_subjects != fbcsp_subjects:

        raise ValueError(
            "\nBaseline and FBCSP contain "
            "different subjects."
        )

    print(
        "\n✓ Subject sets match."
    )

    # --------------------------------------------------------
    # Classifiers
    # --------------------------------------------------------

    baseline_classifiers = set(
        baseline["Classifier"]
    )

    fbcsp_classifiers = set(
        fbcsp["Classifier"]
    )

    print(
        "\nBaseline classifiers:"
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
            "\nBaseline and FBCSP contain "
            "different classifiers."
        )

    print(
        "\n✓ Classifier sets match."
    )

    # --------------------------------------------------------
    # Check expected classifiers
    # --------------------------------------------------------

    for classifier in CLASSIFIERS:

        if classifier not in baseline_classifiers:

            raise ValueError(
                f"Missing classifier: "
                f"{classifier}"
            )

    print(
        "✓ All expected classifiers found."
    )


# ============================================================
# PREPARE PAIRED DATA
# ============================================================

def prepare_paired_data(
    baseline,
    fbcsp,
    classifier
):

    # --------------------------------------------------------
    # Select classifier
    # --------------------------------------------------------

    baseline_classifier = baseline[
        baseline["Classifier"] == classifier
    ].copy()

    fbcsp_classifier = fbcsp[
        fbcsp["Classifier"] == classifier
    ].copy()

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    baseline_classifier = (
        baseline_classifier[
            [
                "Subject",
                "Mean Accuracy (%)"
            ]
        ]
    )

    fbcsp_classifier = (
        fbcsp_classifier[
            [
                "Subject",
                "Mean Accuracy (%)"
            ]
        ]
    )

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    baseline_classifier = (
        baseline_classifier.rename(
            columns={
                "Mean Accuracy (%)":
                    "Baseline"
            }
        )
    )

    fbcsp_classifier = (
        fbcsp_classifier.rename(
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

    # --------------------------------------------------------
    # Sort subjects
    # --------------------------------------------------------

    merged = merged.sort_values(
        "Subject"
    )

    # --------------------------------------------------------
    # Calculate improvement
    # --------------------------------------------------------

    merged["Improvement"] = (
        merged["FBCSP"]
        - merged["Baseline"]
    )

    return merged


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def calculate_descriptive_statistics(
    data
):

    baseline_values = (
        data["Baseline"]
        .to_numpy(
            dtype=float
        )
    )

    fbcsp_values = (
        data["FBCSP"]
        .to_numpy(
            dtype=float
        )
    )

    improvement = (
        data["Improvement"]
        .to_numpy(
            dtype=float
        )
    )

    statistics = {

        "baseline_mean":
            np.mean(
                baseline_values
            ),

        "baseline_std":
            np.std(
                baseline_values,
                ddof=1
            ),

        "fbcsp_mean":
            np.mean(
                fbcsp_values
            ),

        "fbcsp_std":
            np.std(
                fbcsp_values,
                ddof=1
            ),

        "improvement_mean":
            np.mean(
                improvement
            ),

        "improvement_std":
            np.std(
                improvement,
                ddof=1
            ),

        "minimum_improvement":
            np.min(
                improvement
            ),

        "maximum_improvement":
            np.max(
                improvement
            )
    }

    return statistics


# ============================================================
# PAIRED T-TEST
# ============================================================

def run_paired_t_test(
    baseline_values,
    fbcsp_values
):

    result = ttest_rel(
        fbcsp_values,
        baseline_values
    )

    return (
        result.statistic,
        result.pvalue
    )


# ============================================================
# WILCOXON SIGNED-RANK TEST
# ============================================================

def run_wilcoxon_test(
    baseline_values,
    fbcsp_values
):

    differences = (
        fbcsp_values
        - baseline_values
    )

    # --------------------------------------------------------
    # If every difference is exactly zero,
    # Wilcoxon cannot be performed.
    # --------------------------------------------------------

    if np.allclose(
        differences,
        0
    ):

        return (
            0.0,
            1.0
        )

    result = wilcoxon(
        fbcsp_values,
        baseline_values,
        alternative="two-sided"
    )

    return (
        result.statistic,
        result.pvalue
    )


# ============================================================
# SHAPIRO-WILK TEST
# ============================================================

def run_shapiro_test(
    differences
):

    # Shapiro requires at least 3 samples.
    if len(differences) < 3:

        return (
            np.nan,
            np.nan
        )

    result = shapiro(
        differences
    )

    return (
        result.statistic,
        result.pvalue
    )


# ============================================================
# EFFECT SIZE
# ============================================================

def calculate_cohens_d(
    differences
):

    mean_difference = np.mean(
        differences
    )

    std_difference = np.std(
        differences,
        ddof=1
    )

    if std_difference == 0:

        if mean_difference == 0:
            return 0.0

        return np.inf

    return (
        mean_difference
        / std_difference
    )


# ============================================================
# INTERPRET COHEN'S D
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
# INTERPRET P-VALUE
# ============================================================

def interpret_p_value(
    p
):

    if np.isnan(p):

        return "not available"

    if p < 0.001:

        return "highly significant"

    elif p < 0.01:

        return "very significant"

    elif p < 0.05:

        return "statistically significant"

    else:

        return "not statistically significant"


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def analyze_classifier(
    baseline,
    fbcsp,
    classifier
):

    print_section(
        f"STATISTICAL ANALYSIS - {classifier}"
    )

    # --------------------------------------------------------
    # Prepare paired data
    # --------------------------------------------------------

    data = prepare_paired_data(
        baseline,
        fbcsp,
        classifier
    )

    print(
        "\nSubject-level results:"
    )

    print(
        data.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Values
    # --------------------------------------------------------

    baseline_values = (
        data["Baseline"]
        .to_numpy(
            dtype=float
        )
    )

    fbcsp_values = (
        data["FBCSP"]
        .to_numpy(
            dtype=float
        )
    )

    differences = (
        fbcsp_values
        - baseline_values
    )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    stats = (
        calculate_descriptive_statistics(
            data
        )
    )

    print_subsection(
        "DESCRIPTIVE STATISTICS"
    )

    print(
        f"Subjects: "
        f"{len(data)}"
    )

    print(
        f"\nStandard CSP:"
    )

    print(
        f"Mean: "
        f"{stats['baseline_mean']:.2f}%"
    )

    print(
        f"Std: "
        f"{stats['baseline_std']:.2f}%"
    )

    print(
        f"\nFBCSP:"
    )

    print(
        f"Mean: "
        f"{stats['fbcsp_mean']:.2f}%"
    )

    print(
        f"Std: "
        f"{stats['fbcsp_std']:.2f}%"
    )

    print(
        f"\nMean improvement:"
        f" {stats['improvement_mean']:+.2f}"
        f" percentage points"
    )

    print(
        f"Improvement Std:"
        f" {stats['improvement_std']:.2f}"
    )

    print(
        f"Minimum improvement:"
        f" {stats['minimum_improvement']:+.2f}"
    )

    print(
        f"Maximum improvement:"
        f" {stats['maximum_improvement']:+.2f}"
    )

    # --------------------------------------------------------
    # Shapiro-Wilk
    # --------------------------------------------------------

    print_subsection(
        "NORMALITY TEST"
    )

    shapiro_stat, shapiro_p = (
        run_shapiro_test(
            differences
        )
    )

    print(
        f"Shapiro-Wilk statistic:"
        f" {shapiro_stat:.4f}"
    )

    print(
        f"Shapiro-Wilk p-value:"
        f" {shapiro_p:.4f}"
    )

    if np.isnan(shapiro_p):

        print(
            "Normality test unavailable."
        )

    elif shapiro_p >= ALPHA:

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
            "Wilcoxon signed-rank test "
            "is preferred."
        )

    # --------------------------------------------------------
    # Paired t-test
    # --------------------------------------------------------

    print_subsection(
        "PAIRED T-TEST"
    )

    t_statistic, t_pvalue = (
        run_paired_t_test(
            baseline_values,
            fbcsp_values
        )
    )

    print(
        f"t-statistic:"
        f" {t_statistic:.4f}"
    )

    print(
        f"p-value:"
        f" {t_pvalue:.6f}"
    )

    print(
        f"Result:"
        f" {interpret_p_value(t_pvalue)}"
    )

    # --------------------------------------------------------
    # Wilcoxon
    # --------------------------------------------------------

    print_subsection(
        "WILCOXON SIGNED-RANK TEST"
    )

    w_statistic, w_pvalue = (
        run_wilcoxon_test(
            baseline_values,
            fbcsp_values
        )
    )

    print(
        f"Wilcoxon statistic:"
        f" {w_statistic:.4f}"
    )

    print(
        f"p-value:"
        f" {w_pvalue:.6f}"
    )

    print(
        f"Result:"
        f" {interpret_p_value(w_pvalue)}"
    )

    # --------------------------------------------------------
    # Cohen's d
    # --------------------------------------------------------

    print_subsection(
        "EFFECT SIZE"
    )

    cohens_d = (
        calculate_cohens_d(
            differences
        )
    )

    effect_interpretation = (
        interpret_effect_size(
            cohens_d
        )
    )

    print(
        f"Cohen's d:"
        f" {cohens_d:.4f}"
    )

    print(
        f"Effect size:"
        f" {effect_interpretation}"
    )

    # --------------------------------------------------------
    # Individual subjects
    # --------------------------------------------------------

    print_subsection(
        "SUBJECT-LEVEL IMPROVEMENT"
    )

    for _, row in data.iterrows():

        subject = row[
            "Subject"
        ]

        baseline_accuracy = row[
            "Baseline"
        ]

        fbcsp_accuracy = row[
            "FBCSP"
        ]

        improvement = row[
            "Improvement"
        ]

        if improvement > 0:

            direction = "IMPROVED"

        elif improvement < 0:

            direction = "DECREASED"

        else:

            direction = "UNCHANGED"

        print(
            f"{subject}: "
            f"{baseline_accuracy:.2f}% -> "
            f"{fbcsp_accuracy:.2f}% "
            f"({improvement:+.2f}) "
            f"[{direction}]"
        )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "classifier":
            classifier,

        "subjects":
            len(data),

        "baseline_mean":
            stats["baseline_mean"],

        "baseline_std":
            stats["baseline_std"],

        "fbcsp_mean":
            stats["fbcsp_mean"],

        "fbcsp_std":
            stats["fbcsp_std"],

        "improvement_mean":
            stats["improvement_mean"],

        "improvement_std":
            stats["improvement_std"],

        "t_statistic":
            t_statistic,

        "t_pvalue":
            t_pvalue,

        "wilcoxon_statistic":
            w_statistic,

        "wilcoxon_pvalue":
            w_pvalue,

        "shapiro_statistic":
            shapiro_stat,

        "shapiro_pvalue":
            shapiro_p,

        "cohens_d":
            cohens_d,

        "effect_size":
            effect_interpretation
    }


# ============================================================
# PRINT FINAL STATISTICAL SUMMARY
# ============================================================

def print_final_summary(
    results
):

    print_section(
        "FINAL STATISTICAL SUMMARY"
    )

    print(
        f"{'Classifier':<25}"
        f"{'CSP':>12}"
        f"{'FBCSP':>12}"
        f"{'Change':>12}"
        f"{'t-test p':>15}"
        f"{'Wilcoxon p':>15}"
        f"{'Cohen d':>12}"
    )

    print(
        "-" * 105
    )

    for classifier, data in results.items():

        print(
            f"{classifier:<25}"
            f"{data['baseline_mean']:>10.2f}%"
            f"{data['fbcsp_mean']:>10.2f}%"
            f"{data['improvement_mean']:>+10.2f}"
            f"{data['t_pvalue']:>15.6f}"
            f"{data['wilcoxon_pvalue']:>15.6f}"
            f"{data['cohens_d']:>12.4f}"
        )

    print(
        "=" * 105
    )


# ============================================================
# SAVE STATISTICAL RESULTS
# ============================================================

def save_statistical_results(
    results
):

    output_file = (
        RESULTS_DIR /
        "statistical_comparison.csv"
    )

    rows = []

    for classifier, data in results.items():

        rows.append({

            "Classifier":
                classifier,

            "Standard CSP Mean (%)":
                data["baseline_mean"],

            "Standard CSP Std (%)":
                data["baseline_std"],

            "FBCSP Mean (%)":
                data["fbcsp_mean"],

            "FBCSP Std (%)":
                data["fbcsp_std"],

            "Mean Improvement (percentage points)":
                data["improvement_mean"],

            "Improvement Std":
                data["improvement_std"],

            "Paired t-test statistic":
                data["t_statistic"],

            "Paired t-test p-value":
                data["t_pvalue"],

            "Wilcoxon statistic":
                data["wilcoxon_statistic"],

            "Wilcoxon p-value":
                data["wilcoxon_pvalue"],

            "Shapiro-Wilk statistic":
                data["shapiro_statistic"],

            "Shapiro-Wilk p-value":
                data["shapiro_pvalue"],

            "Cohen's d":
                data["cohens_d"],

            "Effect Size":
                data["effect_size"]
        })

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nStatistical results saved to:"
        f"\n{output_file}"
    )


# ============================================================
# PLOT BASELINE VS FBCSP
# ============================================================

def plot_comparison(
    baseline,
    fbcsp
):

    print_section(
        "GENERATING STATISTICAL COMPARISON PLOT"
    )

    classifiers = (
        CLASSIFIERS
    )

    baseline_means = []
    fbcsp_means = []

    baseline_stds = []
    fbcsp_stds = []

    for classifier in classifiers:

        baseline_data = (
            baseline[
                baseline["Classifier"]
                == classifier
            ]["Mean Accuracy (%)"]
            .to_numpy(
                dtype=float
            )
        )

        fbcsp_data = (
            fbcsp[
                fbcsp["Classifier"]
                == classifier
            ]["Mean Accuracy (%)"]
            .to_numpy(
                dtype=float
            )
        )

        baseline_means.append(
            np.mean(
                baseline_data
            )
        )

        fbcsp_means.append(
            np.mean(
                fbcsp_data
            )
        )

        baseline_stds.append(
            np.std(
                baseline_data,
                ddof=1
            )
        )

        fbcsp_stds.append(
            np.std(
                fbcsp_data,
                ddof=1
            )
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    x = np.arange(
        len(classifiers)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    baseline_bars = ax.bar(
        x - width / 2,
        baseline_means,
        width,
        yerr=baseline_stds,
        capsize=5,
        label="Standard CSP"
    )

    fbcsp_bars = ax.bar(
        x + width / 2,
        fbcsp_means,
        width,
        yerr=fbcsp_stds,
        capsize=5,
        label="FBCSP"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    for bar, value in zip(
        baseline_bars,
        baseline_means
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height()
            + 1,

            f"{value:.2f}%",

            ha="center",
            va="bottom"
        )

    for bar, value in zip(
        fbcsp_bars,
        fbcsp_means
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height()
            + 1,

            f"{value:.2f}%",

            ha="center",
            va="bottom"
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        "Standard CSP vs FBCSP",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Classifier",
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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        RESULTS_DIR /
        "statistical_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\nPlot saved to:"
        f"\n{output_file}"
    )

    return fig


# ============================================================
# PRINT RESEARCH INTERPRETATION
# ============================================================

def print_research_interpretation(
    results
):

    print_section(
        "RESEARCH INTERPRETATION"
    )

    for classifier, data in results.items():

        improvement = (
            data["improvement_mean"]
        )

        p_value = (
            data["wilcoxon_pvalue"]
        )

        d = (
            data["cohens_d"]
        )

        print(
            f"\n{classifier}"
        )

        print(
            "-" * 50
        )

        if improvement > 0:

            print(
                f"FBCSP improved mean "
                f"accuracy by "
                f"{improvement:.2f} "
                f"percentage points."
            )

        elif improvement < 0:

            print(
                f"FBCSP decreased mean "
                f"accuracy by "
                f"{abs(improvement):.2f} "
                f"percentage points."
            )

        else:

            print(
                "FBCSP produced no "
                "change in mean accuracy."
            )

        print(
            f"Wilcoxon p-value: "
            f"{p_value:.6f}"
        )

        print(
            f"Cohen's d: "
            f"{d:.4f}"
        )

        if p_value < ALPHA:

            print(
                "Conclusion: The difference "
                "is statistically significant "
                f"(p < {ALPHA})."
            )

        else:

            print(
                "Conclusion: The difference "
                "is NOT statistically "
                f"significant (p >= {ALPHA})."
            )

        print(
            f"Effect size interpretation: "
            f"{data['effect_size']}."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "EIDOS-BCI\n"
        "STATISTICAL COMPARISON\n"
        "STANDARD CSP vs FBCSP"
    )

    print(
        "\nSignificance level:"
        f" alpha = {ALPHA}"
    )

    print(
        "\nDataset:"
        "\nBCI Competition IV Dataset 2a"
    )

    print(
        "\nStatistical unit:"
        "\nSubject"
    )

    print(
        "\nComparison:"
        "\nStandard CSP vs FBCSP"
    )

    print(
        "\nTests:"
        "\nPaired t-test"
        "\nWilcoxon signed-rank test"
        "\nShapiro-Wilk normality test"
        "\nCohen's d effect size"
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

    validate_results(
        baseline,
        fbcsp
    )

    # ========================================================
    # ANALYZE
    # ========================================================

    statistical_results = {}

    for classifier in CLASSIFIERS:

        result = (
            analyze_classifier(
                baseline,
                fbcsp,
                classifier
            )
        )

        statistical_results[
            classifier
        ] = result

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_final_summary(
        statistical_results
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print_research_interpretation(
        statistical_results
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_statistical_results(
        statistical_results
    )

    # ========================================================
    # PLOT
    # ========================================================

    plot_comparison(
        baseline,
        fbcsp
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 100)

    print(
        "\nGenerated files:"
    )

    print(
        f"\n1. {RESULTS_DIR / 'statistical_comparison.csv'}"
    )

    print(
        f"\n2. {RESULTS_DIR / 'statistical_comparison.png'}"
    )

    print(
        "\nNext research stage:"
    )

    print(
        "Four-class motor imagery classification"
    )

    print("=" * 100)

    plt.show()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()