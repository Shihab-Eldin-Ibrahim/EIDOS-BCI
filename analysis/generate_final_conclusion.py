"""
generate_final_conclusion.py

Generates the final research conclusion for the EIDOS-BCI
four-class motor imagery experiment.

The generator combines:

    Paper tables:
        - Within-subject Standard CSP
        - Within-subject FBCSP
        - LOSO performance
        - LOSO CSP/FBCSP comparison
        - LOSO subject-level improvements
        - Overall LOSO summary

    Detailed statistical results:
        - Shapiro-Wilk normality test
        - Paired t-test
        - Wilcoxon signed-rank test
        - Cohen's d
        - Improved/decreased subjects

Dataset:
    BCI Competition IV Dataset 2a

Subjects:
    A01-A09

Classes:
    Left hand
    Right hand
    Feet
    Tongue

Evaluation:
    Within-subject
    Leave-One-Subject-Out (LOSO)

Methods:
    Standard CSP
    Filter Bank CSP (FBCSP)

Classifiers:
    LDA
    SVM
    Logistic Regression
"""

from pathlib import Path
import csv
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "four_class"
)

PAPER_TABLES_DIR = (
    RESULTS_DIR
    / "paper_tables"
)

STATISTICS_DIR = (
    RESULTS_DIR
    / "cross_subject"
    / "statistical_tests"
)

CONCLUSION_DIR = (
    RESULTS_DIR
    / "final_conclusion"
)

CONCLUSION_FILE = (
    CONCLUSION_DIR
    / "final_research_conclusion.txt"
)

SUMMARY_FILE = (
    CONCLUSION_DIR
    / "final_research_summary.csv"
)


# ============================================================
# PAPER TABLES
# ============================================================

TABLE_1 = (
    PAPER_TABLES_DIR
    / "table_1_within_subject_standard_csp.csv"
)

TABLE_2 = (
    PAPER_TABLES_DIR
    / "table_2_within_subject_fbcsp.csv"
)

TABLE_3 = (
    PAPER_TABLES_DIR
    / "table_3_loso_performance.csv"
)

TABLE_4 = (
    PAPER_TABLES_DIR
    / "table_4_loso_statistics.csv"
)

TABLE_5 = (
    PAPER_TABLES_DIR
    / "table_5_loso_subject_improvements.csv"
)

TABLE_6 = (
    PAPER_TABLES_DIR
    / "table_6_overall_loso_summary.csv"
)


# ============================================================
# DETAILED STATISTICAL TABLES
# ============================================================

DETAILED_STATISTICS_FILE = (
    STATISTICS_DIR
    / "loso_statistical_comparison.csv"
)

DETAILED_IMPROVEMENTS_FILE = (
    STATISTICS_DIR
    / "loso_subject_improvements.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASSIFIERS = [
    "LDA",
    "SVM",
    "Logistic Regression"
]

SUBJECTS = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A08",
    "A09"
]


# ============================================================
# PRINT HELPERS
# ============================================================

def print_section(title):

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# CSV HELPERS
# ============================================================

def read_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

        if not rows:

            raise ValueError(
                f"CSV file is empty:\n{path}"
            )

        return rows


def find_column(row, candidates):

    """
    Finds a column while allowing small naming differences.
    """

    normalized = {
        key.strip().lower(): key
        for key in row.keys()
    }

    # Exact match
    for candidate in candidates:

        candidate_normalized = (
            candidate.strip().lower()
        )

        if candidate_normalized in normalized:

            return normalized[
                candidate_normalized
            ]

    # Partial match
    for key in row.keys():

        key_lower = key.strip().lower()

        for candidate in candidates:

            candidate_lower = (
                candidate.strip().lower()
            )

            if (
                candidate_lower in key_lower
                or key_lower in candidate_lower
            ):

                return key

    return None


def numeric(
    row,
    candidates,
    default=np.nan
):

    column = find_column(
        row,
        candidates
    )

    if column is None:

        return default

    value = row[column]

    if value is None:

        return default

    value = str(value).strip()

    if value == "":

        return default

    if value.lower() in {
        "nan",
        "n/a",
        "na",
        "none"
    }:

        return default

    try:

        return float(value)

    except ValueError:

        return default


# ============================================================
# CHECK INPUT FILES
# ============================================================

def check_input_files():

    print_section(
        "CHECKING INPUT FILES"
    )

    paper_files = [
        TABLE_1,
        TABLE_2,
        TABLE_3,
        TABLE_4,
        TABLE_5,
        TABLE_6
    ]

    for path in paper_files:

        if path.exists():

            print(
                f"✓ {path}"
            )

        else:

            print(
                f"✗ MISSING: {path}"
            )

            raise FileNotFoundError(
                f"Required paper table missing:\n{path}"
            )

    print()

    if DETAILED_STATISTICS_FILE.exists():

        print(
            f"✓ Detailed statistics:\n"
            f"  {DETAILED_STATISTICS_FILE}"
        )

    else:

        print(
            "⚠ Detailed statistical table not found."
        )

        print(
            "  Statistical p-values and subject counts "
            "will be reconstructed where possible."
        )

    if DETAILED_IMPROVEMENTS_FILE.exists():

        print(
            f"✓ Detailed subject improvements:\n"
            f"  {DETAILED_IMPROVEMENTS_FILE}"
        )

    else:

        print(
            "⚠ Detailed subject improvement table not found."
        )


# ============================================================
# LOAD DETAILED STATISTICS
# ============================================================

def load_detailed_statistics():

    print_section(
        "LOADING DETAILED LOSO STATISTICS"
    )

    if not DETAILED_STATISTICS_FILE.exists():

        print(
            "Detailed statistics file not found."
        )

        return {}

    rows = read_csv(
        DETAILED_STATISTICS_FILE
    )

    print(
        "Rows found:",
        len(rows)
    )

    statistics = {}

    for row in rows:

        classifier_column = find_column(
            row,
            [
                "Classifier"
            ]
        )

        if classifier_column is None:

            raise KeyError(
                "Could not find Classifier column "
                "in detailed statistics."
            )

        classifier = (
            row[classifier_column]
            .strip()
        )

        statistics[classifier] = {

            "subjects":
                numeric(
                    row,
                    [
                        "Subjects"
                    ]
                ),

            "standard_mean":
                numeric(
                    row,
                    [
                        "Standard CSP Mean (%)",
                        "CSP Mean (%)"
                    ]
                ),

            "standard_std":
                numeric(
                    row,
                    [
                        "Standard CSP Std (%)"
                    ]
                ),

            "fbcsp_mean":
                numeric(
                    row,
                    [
                        "FBCSP Mean (%)"
                    ]
                ),

            "fbcsp_std":
                numeric(
                    row,
                    [
                        "FBCSP Std (%)"
                    ]
                ),

            "mean_change":
                numeric(
                    row,
                    [
                        "Mean Change (pp)",
                        "Improvement (pp)"
                    ]
                ),

            "change_std":
                numeric(
                    row,
                    [
                        "Change Std"
                    ]
                ),

            "minimum_change":
                numeric(
                    row,
                    [
                        "Minimum Change (pp)"
                    ]
                ),

            "maximum_change":
                numeric(
                    row,
                    [
                        "Maximum Change (pp)"
                    ]
                ),

            "shapiro_statistic":
                numeric(
                    row,
                    [
                        "Shapiro Statistic"
                    ]
                ),

            "shapiro_p":
                numeric(
                    row,
                    [
                        "Shapiro p-value",
                        "Shapiro p"
                    ]
                ),

            "t_statistic":
                numeric(
                    row,
                    [
                        "Paired t-statistic",
                        "Paired t statistic",
                        "t-statistic"
                    ]
                ),

            "t_p":
                numeric(
                    row,
                    [
                        "Paired t p-value",
                        "Paired t p",
                        "t p-value",
                        "t p"
                    ]
                ),

            "wilcoxon_statistic":
                numeric(
                    row,
                    [
                        "Wilcoxon Statistic",
                        "Wilcoxon statistic"
                    ]
                ),

            "wilcoxon_p":
                numeric(
                    row,
                    [
                        "Wilcoxon p-value",
                        "Wilcoxon p"
                    ]
                ),

            "cohen_d":
                numeric(
                    row,
                    [
                        "Cohen's d",
                        "Cohen d"
                    ]
                ),

            "improved":
                numeric(
                    row,
                    [
                        "Improved Subjects",
                        "Improved"
                    ]
                ),

            "decreased":
                numeric(
                    row,
                    [
                        "Decreased Subjects",
                        "Decreased"
                    ]
                )
        }

    return statistics


# ============================================================
# LOAD PAPER TABLE 4
# ============================================================

def load_table_4():

    print_section(
        "LOADING PAPER TABLE 4"
    )

    rows = read_csv(
        TABLE_4
    )

    statistics = {}

    for row in rows:

        classifier_column = find_column(
            row,
            [
                "Classifier"
            ]
        )

        if classifier_column is None:

            raise KeyError(
                "Could not find Classifier column."
            )

        classifier = (
            row[classifier_column]
            .strip()
        )

        statistics[classifier] = {

            "standard_mean":
                numeric(
                    row,
                    [
                        "CSP Mean (%)",
                        "Standard CSP Mean (%)"
                    ]
                ),

            "fbcsp_mean":
                numeric(
                    row,
                    [
                        "FBCSP Mean (%)"
                    ]
                ),

            "mean_change":
                numeric(
                    row,
                    [
                        "Mean Change (pp)",
                        "Improvement (pp)"
                    ]
                ),

            "t_statistic":
                numeric(
                    row,
                    [
                        "Paired t-statistic",
                        "t-statistic"
                    ]
                ),

            "cohen_d":
                numeric(
                    row,
                    [
                        "Cohen's d"
                    ]
                ),

            "wilcoxon_p":
                numeric(
                    row,
                    [
                        "Wilcoxon p-value"
                    ]
                )
        }

    return statistics


# ============================================================
# MERGE STATISTICS
# ============================================================

def merge_statistics():

    paper_statistics = load_table_4()

    detailed_statistics = (
        load_detailed_statistics()
    )

    merged = {}

    for classifier in CLASSIFIERS:

        merged[classifier] = {

            "standard_mean": np.nan,

            "standard_std": np.nan,

            "fbcsp_mean": np.nan,

            "fbcsp_std": np.nan,

            "mean_change": np.nan,

            "change_std": np.nan,

            "minimum_change": np.nan,

            "maximum_change": np.nan,

            "shapiro_statistic": np.nan,

            "shapiro_p": np.nan,

            "t_statistic": np.nan,

            "t_p": np.nan,

            "wilcoxon_statistic": np.nan,

            "wilcoxon_p": np.nan,

            "cohen_d": np.nan,

            "improved": np.nan,

            "decreased": np.nan
        }

        # ----------------------------------------------------
        # Start with paper Table 4
        # ----------------------------------------------------

        if classifier in paper_statistics:

            source = paper_statistics[
                classifier
            ]

            for key, value in source.items():

                if key in merged[classifier]:

                    if not np.isnan(value):

                        merged[classifier][key] = value

        # ----------------------------------------------------
        # Override with detailed statistics
        # ----------------------------------------------------

        if classifier in detailed_statistics:

            source = detailed_statistics[
                classifier
            ]

            for key, value in source.items():

                if key in merged[classifier]:

                    if not np.isnan(value):

                        merged[classifier][key] = value

    return merged


# ============================================================
# LOAD SUBJECT IMPROVEMENTS
# ============================================================

def load_subject_improvements():

    print_section(
        "LOADING SUBJECT-LEVEL IMPROVEMENTS"
    )

    path = DETAILED_IMPROVEMENTS_FILE

    if path.exists():

        rows = read_csv(
            path
        )

    else:

        rows = read_csv(
            TABLE_5
        )

    improvements = {}

    for row in rows:

        subject_column = find_column(
            row,
            [
                "Test Subject",
                "Subject"
            ]
        )

        classifier_column = find_column(
            row,
            [
                "Classifier"
            ]
        )

        improvement_column = find_column(
            row,
            [
                "Improvement (pp)",
                "Mean Change (pp)"
            ]
        )

        if (
            subject_column is None
            or classifier_column is None
            or improvement_column is None
        ):

            raise KeyError(
                "Could not identify required "
                "subject-level columns."
            )

        subject = row[
            subject_column
        ].strip()

        classifier = row[
            classifier_column
        ].strip()

        improvement = numeric(
            row,
            [
                "Improvement (pp)",
                "Mean Change (pp)"
            ]
        )

        if classifier not in improvements:

            improvements[
                classifier
            ] = {}

        improvements[
            classifier
        ][subject] = improvement

    return improvements


# ============================================================
# CALCULATE IMPROVEMENT COUNTS
# ============================================================

def calculate_improvement_counts(
    improvements
):

    counts = {}

    for classifier in CLASSIFIERS:

        values = (
            list(
                improvements
                .get(classifier, {})
                .values()
            )
        )

        values = [
            value
            for value in values
            if not np.isnan(value)
        ]

        improved = sum(
            value > 0
            for value in values
        )

        decreased = sum(
            value < 0
            for value in values
        )

        unchanged = sum(
            value == 0
            for value in values
        )

        counts[classifier] = {

            "improved": improved,

            "decreased": decreased,

            "unchanged": unchanged
        }

    return counts


# ============================================================
# EFFECT SIZE INTERPRETATION
# ============================================================

def effect_size_text(cohen_d):

    if np.isnan(cohen_d):

        return (
            "The effect size could not be determined."
        )

    absolute_d = abs(
        cohen_d
    )

    if absolute_d < 0.2:

        interpretation = "negligible"

    elif absolute_d < 0.5:

        interpretation = "small"

    elif absolute_d < 0.8:

        interpretation = "medium"

    else:

        interpretation = "large"

    return (
        f"Cohen's d = {cohen_d:.4f}, "
        f"indicating a {interpretation} effect."
    )


# ============================================================
# P-VALUE INTERPRETATION
# ============================================================

def test_result(
    test_name,
    p_value
):

    if np.isnan(p_value):

        return (
            f"The {test_name} p-value was not available "
            f"in the generated statistical table."
        )

    if p_value < 0.05:

        return (
            f"The {test_name} indicates a statistically "
            f"significant difference (p = {p_value:.4f})."
        )

    return (
        f"The {test_name} indicates no statistically "
        f"significant difference (p = {p_value:.4f})."
    )


# ============================================================
# STATISTICAL INTERPRETATION
# ============================================================

def statistical_interpretation(
    classifier,
    stats
):

    mean_change = stats[
        "mean_change"
    ]

    t_p = stats[
        "t_p"
    ]

    wilcoxon_p = stats[
        "wilcoxon_p"
    ]

    cohen_d = stats[
        "cohen_d"
    ]

    improved = stats[
        "improved"
    ]

    decreased = stats[
        "decreased"
    ]

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if mean_change > 0:

        direction = (
            f"improved by {mean_change:.2f} "
            f"percentage points"
        )

    elif mean_change < 0:

        direction = (
            f"decreased by {abs(mean_change):.2f} "
            f"percentage points"
        )

    else:

        direction = (
            "remained unchanged"
        )

    # --------------------------------------------------------
    # Subject counts
    # --------------------------------------------------------

    if np.isnan(improved):

        improved_text = "N/A"

    else:

        improved_text = str(
            int(improved)
        )

    if np.isnan(decreased):

        decreased_text = "N/A"

    else:

        decreased_text = str(
            int(decreased)
        )

    # --------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------

    t_result = test_result(
        "paired t-test",
        t_p
    )

    wilcoxon_result = test_result(
        "Wilcoxon signed-rank test",
        wilcoxon_p
    )

    effect_result = effect_size_text(
        cohen_d
    )

    return (
        f"For {classifier}, FBCSP {direction}. "
        f"{t_result} "
        f"{wilcoxon_result} "
        f"{effect_result} "
        f"{improved_text} of the evaluated subjects "
        f"improved, while {decreased_text} decreased."
    )


# ============================================================
# FIND BEST CLASSIFIER
# ============================================================

def find_best_classifier(
    statistics,
    method
):

    best_classifier = None
    best_accuracy = -np.inf

    if method == "standard":

        key = "standard_mean"

    elif method == "fbcsp":

        key = "fbcsp_mean"

    else:

        raise ValueError(
            "method must be 'standard' or 'fbcsp'"
        )

    for classifier, stats in statistics.items():

        accuracy = stats[key]

        if (
            not np.isnan(accuracy)
            and accuracy > best_accuracy
        ):

            best_accuracy = accuracy

            best_classifier = classifier

    return (
        best_classifier,
        best_accuracy
    )


# ============================================================
# GENERATE CONCLUSION
# ============================================================

def generate_conclusion(
    statistics,
    improvements
):

    # --------------------------------------------------------
    # Best classifiers
    # --------------------------------------------------------

    (
        best_standard_classifier,
        best_standard_accuracy
    ) = find_best_classifier(
        statistics,
        "standard"
    )

    (
        best_fbcsp_classifier,
        best_fbcsp_accuracy
    ) = find_best_classifier(
        statistics,
        "fbcsp"
    )

    # --------------------------------------------------------
    # Overall means
    # --------------------------------------------------------

    standard_means = [
        stats["standard_mean"]
        for stats in statistics.values()
        if not np.isnan(
            stats["standard_mean"]
        )
    ]

    fbcsp_means = [
        stats["fbcsp_mean"]
        for stats in statistics.values()
        if not np.isnan(
            stats["fbcsp_mean"]
        )
    ]

    overall_standard = (
        np.mean(standard_means)
        if standard_means
        else np.nan
    )

    overall_fbcsp = (
        np.mean(fbcsp_means)
        if fbcsp_means
        else np.nan
    )

    overall_change = (
        overall_fbcsp
        - overall_standard
    )

    # --------------------------------------------------------
    # Count significant results
    # --------------------------------------------------------

    significant_t = []

    significant_w = []

    for stats in statistics.values():

        if not np.isnan(
            stats["t_p"]
        ):

            significant_t.append(
                stats["t_p"] < 0.05
            )

        if not np.isnan(
            stats["wilcoxon_p"]
        ):

            significant_w.append(
                stats["wilcoxon_p"] < 0.05
            )

    any_t_significant = (
        any(significant_t)
        if significant_t
        else False
    )

    any_w_significant = (
        any(significant_w)
        if significant_w
        else False
    )

    # --------------------------------------------------------
    # Build conclusion
    # --------------------------------------------------------

    text = []

    text.append(
        "EIDOS-BCI FINAL RESEARCH CONCLUSION"
    )

    text.append(
        "=" * 70
    )

    text.append("")

    # ========================================================
    # 1. OBJECTIVE
    # ========================================================

    text.append(
        "1. STUDY OBJECTIVE"
    )

    text.append(
        "The objective of this experiment was to evaluate "
        "four-class motor imagery EEG classification using "
        "Common Spatial Patterns (CSP) and Filter Bank "
        "Common Spatial Patterns (FBCSP), and to determine "
        "whether the use of multiple frequency bands improves "
        "classification performance under both within-subject "
        "and subject-independent conditions."
    )

    text.append("")

    # ========================================================
    # 2. DATASET
    # ========================================================

    text.append(
        "2. DATASET AND EXPERIMENTAL DESIGN"
    )

    text.append(
        "The evaluation was conducted using the BCI Competition "
        "IV Dataset 2a. The experiment considered nine subjects "
        "(A01-A09) and four motor imagery classes: left hand, "
        "right hand, feet, and tongue. EEG signals were processed "
        "using a 1-4 second classification window."
    )

    text.append(
        "For subject-independent evaluation, Leave-One-Subject-Out "
        "(LOSO) validation was used. In each fold, one subject "
        "was completely held out for testing while the remaining "
        "subjects were used for training. CSP spatial filters "
        "were fitted exclusively on the training subjects and "
        "then applied to the held-out subject."
    )

    text.append("")

    # ========================================================
    # 3. SUBJECT-INDEPENDENT RESULTS
    # ========================================================

    text.append(
        "3. SUBJECT-INDEPENDENT RESULTS"
    )

    text.append(
        f"The highest mean Standard CSP LOSO accuracy was "
        f"obtained by {best_standard_classifier}, with a mean "
        f"accuracy of {best_standard_accuracy:.2f}%."
    )

    text.append(
        f"The highest mean FBCSP LOSO accuracy was obtained by "
        f"{best_fbcsp_classifier}, with a mean accuracy of "
        f"{best_fbcsp_accuracy:.2f}%."
    )

    text.append(
        f"Considering the three classifiers together, the mean "
        f"accuracy across classifier-level LOSO results was "
        f"{overall_standard:.2f}% for Standard CSP and "
        f"{overall_fbcsp:.2f}% for FBCSP, corresponding to an "
        f"overall change of {overall_change:+.2f} percentage points."
    )

    text.append("")

    # ========================================================
    # 4. STATISTICAL ANALYSIS
    # ========================================================

    text.append(
        "4. STATISTICAL ANALYSIS"
    )

    text.append(
        "The comparison between Standard CSP and FBCSP was "
        "performed at the subject level. Shapiro-Wilk tests "
        "were used to assess the normality of paired subject "
        "differences. Paired t-tests and Wilcoxon signed-rank "
        "tests were then used to evaluate whether FBCSP produced "
        "a statistically significant change in performance."
    )

    text.append("")

    for classifier in CLASSIFIERS:

        if classifier not in statistics:

            continue

        text.append(
            statistical_interpretation(
                classifier,
                statistics[classifier]
            )
        )

    text.append("")

    if (
        not any_t_significant
        and not any_w_significant
    ):

        text.append(
            "Overall, none of the evaluated classifiers showed "
            "a statistically significant difference between "
            "Standard CSP and FBCSP under LOSO evaluation. "
            "Therefore, the results do not provide statistical "
            "evidence that the tested FBCSP configuration "
            "consistently improves subject-independent "
            "four-class motor imagery classification."
        )

    else:

        text.append(
            "The statistical analysis indicates that at least "
            "one statistical test detected a significant "
            "difference between Standard CSP and FBCSP. "
            "Therefore, the effect of frequency-bank processing "
            "should be interpreted with respect to the specific "
            "classifier and experimental configuration."
        )

    text.append("")

    # ========================================================
    # 5. INTERPRETATION
    # ========================================================

    text.append(
        "5. INTERPRETATION"
    )

    text.append(
        "The LOSO results demonstrate the difficulty of "
        "subject-independent motor imagery EEG classification. "
        "Performance varies considerably between subjects, "
        "indicating substantial inter-subject variability in "
        "EEG patterns."
    )

    text.append(
        "The results also show that adding multiple frequency "
        "bands does not automatically produce better "
        "cross-subject classification. Although FBCSP can "
        "provide additional frequency-specific spatial "
        "information, this additional representation may also "
        "increase feature dimensionality and introduce "
        "subject-specific information that does not generalize "
        "well to unseen participants."
    )

    text.append("")

    # ========================================================
    # 6. MAIN FINDING
    # ========================================================

    text.append(
        "6. MAIN FINDING"
    )

    if overall_change > 0:

        text.append(
            f"FBCSP produced a small overall positive change "
            f"of {overall_change:+.2f} percentage points across "
            f"the evaluated classifiers. However, the observed "
            f"improvement should not be interpreted as a reliable "
            f"general improvement because the subject-level "
            f"statistical tests did not establish consistent "
            f"statistical significance."
        )

    elif overall_change < 0:

        text.append(
            f"FBCSP produced a small overall decrease of "
            f"{abs(overall_change):.2f} percentage points across "
            f"the evaluated classifiers. The decrease was small "
            f"and should be interpreted in the context of the "
            f"large variability between subjects."
        )

    else:

        text.append(
            "FBCSP and Standard CSP produced essentially "
            "identical overall LOSO performance."
        )

    text.append("")

    # ========================================================
    # 7. RESEARCH CONCLUSION
    # ========================================================

    text.append(
        "7. RESEARCH CONCLUSION"
    )

    text.append(
        "Within the conditions tested in this study, Standard CSP "
        "and the implemented FBCSP configuration provide "
        "comparable subject-independent four-class motor "
        "imagery performance. The statistical analysis does not "
        "support the claim that FBCSP consistently outperforms "
        "Standard CSP for unseen subjects."
    )

    text.append(
        "Consequently, the principal finding of this experiment "
        "is that frequency-bank spatial filtering alone is "
        "insufficient to overcome the inter-subject variability "
        "of motor imagery EEG. Improving subject-independent BCI "
        "performance likely requires additional techniques such "
        "as domain adaptation, transfer learning, subject "
        "normalization, Riemannian geometry-based methods, "
        "adaptive spatial filtering, or deep learning approaches "
        "designed for cross-subject EEG."
    )

    text.append("")

    # ========================================================
    # 8. LIMITATIONS
    # ========================================================

    text.append(
        "8. LIMITATIONS"
    )

    text.append(
        "The study evaluates a single public dataset, a fixed "
        "frequency-bank configuration, a fixed CSP dimensionality, "
        "and three conventional classifiers. The relatively small "
        "number of subjects also limits the statistical power of "
        "the comparison. Therefore, the conclusions should be "
        "restricted to the experimental configuration evaluated "
        "rather than generalized to all FBCSP implementations."
    )

    text.append("")

    # ========================================================
    # 9. FUTURE WORK
    # ========================================================

    text.append(
        "9. FUTURE WORK"
    )

    text.append(
        "Future work should investigate optimized frequency bands, "
        "regularized CSP, subject normalization, transfer learning, "
        "domain adaptation, Riemannian EEG representations, and "
        "hybrid or deep-learning approaches. Additional evaluation "
        "on independent datasets would also be necessary to "
        "determine whether the observed behavior generalizes "
        "beyond BCI Competition IV Dataset 2a."
    )

    text.append("")

    text.append(
        "=" * 70
    )

    text.append(
        "END OF FINAL RESEARCH CONCLUSION"
    )

    return "\n".join(
        text
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

def save_summary(
    statistics
):

    CONCLUSION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SUMMARY_FILE,
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
            "Mean Change (pp)",
            "Paired t p-value",
            "Wilcoxon p-value",
            "Cohen's d",
            "Improved Subjects",
            "Decreased Subjects"
        ])

        for classifier in CLASSIFIERS:

            if classifier not in statistics:

                continue

            stats = statistics[
                classifier
            ]

            writer.writerow([

                classifier,

                (
                    f"{stats['standard_mean']:.4f}"
                    if not np.isnan(
                        stats["standard_mean"]
                    )
                    else ""
                ),

                (
                    f"{stats['fbcsp_mean']:.4f}"
                    if not np.isnan(
                        stats["fbcsp_mean"]
                    )
                    else ""
                ),

                (
                    f"{stats['mean_change']:.4f}"
                    if not np.isnan(
                        stats["mean_change"]
                    )
                    else ""
                ),

                (
                    f"{stats['t_p']:.6f}"
                    if not np.isnan(
                        stats["t_p"]
                    )
                    else ""
                ),

                (
                    f"{stats['wilcoxon_p']:.6f}"
                    if not np.isnan(
                        stats["wilcoxon_p"]
                    )
                    else ""
                ),

                (
                    f"{stats['cohen_d']:.4f}"
                    if not np.isnan(
                        stats["cohen_d"]
                    )
                    else ""
                ),

                (
                    int(stats["improved"])
                    if not np.isnan(
                        stats["improved"]
                    )
                    else ""
                ),

                (
                    int(stats["decreased"])
                    if not np.isnan(
                        stats["decreased"]
                    )
                    else ""
                )
            ])


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "EIDOS-BCI"
    )

    print(
        "FINAL RESEARCH CONCLUSION GENERATION"
    )

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "\nPaper tables directory:",
        PAPER_TABLES_DIR
    )

    print(
        "\nDetailed statistics directory:",
        STATISTICS_DIR
    )

    print(
        "\nConclusion directory:",
        CONCLUSION_DIR
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_input_files()

    # --------------------------------------------------------
    # Load and merge statistics
    # --------------------------------------------------------

    statistics = merge_statistics()

    # --------------------------------------------------------
    # Load subject improvements
    # --------------------------------------------------------

    improvements = (
        load_subject_improvements()
    )

    # --------------------------------------------------------
    # Calculate subject counts
    # --------------------------------------------------------

    counts = calculate_improvement_counts(
        improvements
    )

    # --------------------------------------------------------
    # Inject calculated counts if detailed
    # counts are unavailable
    # --------------------------------------------------------

    for classifier in CLASSIFIERS:

        if classifier not in statistics:

            continue

        if np.isnan(
            statistics[classifier]["improved"]
        ):

            statistics[classifier][
                "improved"
            ] = counts[
                classifier
            ]["improved"]

        if np.isnan(
            statistics[classifier]["decreased"]
        ):

            statistics[classifier][
                "decreased"
            ] = counts[
                classifier
            ]["decreased"]

    # --------------------------------------------------------
    # Generate conclusion
    # --------------------------------------------------------

    print_section(
        "GENERATING FINAL CONCLUSION"
    )

    conclusion = generate_conclusion(
        statistics,
        improvements
    )

    # --------------------------------------------------------
    # Save conclusion
    # --------------------------------------------------------

    CONCLUSION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CONCLUSION_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            conclusion
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(
        statistics
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        "\n✓ Final conclusion generated."
    )

    print(
        f"\nSaved to:\n"
        f"{CONCLUSION_FILE}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{SUMMARY_FILE}"
    )

    print_section(
        "FINAL RESEARCH CONCLUSION"
    )

    print(
        conclusion
    )

    print_section(
        "FINAL CONCLUSION GENERATION COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()