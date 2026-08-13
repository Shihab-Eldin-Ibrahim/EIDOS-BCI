"""
generate_final_conclusion.py

Generates the final research conclusion for the EIDOS-BCI
four-class motor imagery experiment.

Uses:
    - Within-subject Standard CSP results
    - Within-subject FBCSP results
    - LOSO Standard CSP results
    - LOSO FBCSP results
    - LOSO statistical tests
    - LOSO subject-level improvements

Dataset:
    BCI Competition IV Dataset 2a

Classes:
    Left hand, Right hand, Feet, Tongue
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
# INPUT FILES
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

    for candidate in candidates:

        candidate_normalized = (
            candidate.strip().lower()
        )

        if candidate_normalized in normalized:

            return normalized[
                candidate_normalized
            ]

    # Fallback: partial matching
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


def numeric(row, candidates, default=np.nan):

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

    files = [
        TABLE_1,
        TABLE_2,
        TABLE_3,
        TABLE_4,
        TABLE_5,
        TABLE_6
    ]

    for path in files:

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


# ============================================================
# LOAD STATISTICAL RESULTS
# ============================================================

def load_statistics():

    print_section(
        "LOADING LOSO STATISTICAL RESULTS"
    )

    rows = read_csv(
        TABLE_4
    )

    print(
        "Rows found:",
        len(rows)
    )

    print(
        "Columns found:"
    )

    for column in rows[0].keys():

        print(
            f"  - {column}"
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
                "Could not find 'Classifier' column."
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
                        "Standard CSP Mean (%)"
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

            # IMPORTANT:
            # Your actual table uses:
            # "Paired t p-value"
            #
            # This explicitly supports that column.
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
                        "Wilcoxon Statistic"
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
# LOAD SUBJECT IMPROVEMENTS
# ============================================================

def load_subject_improvements():

    print_section(
        "LOADING SUBJECT-LEVEL IMPROVEMENTS"
    )

    rows = read_csv(
        TABLE_5
    )

    print(
        "Rows found:",
        len(rows)
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
                "subject-level columns in Table 5."
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
# GENERATE STATISTICAL INTERPRETATION
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

    if (
        not np.isnan(t_p)
        and t_p < 0.05
    ):

        t_result = (
            "The paired t-test indicates "
            "a statistically significant "
            "difference between Standard CSP "
            "and FBCSP."
        )

    else:

        t_result = (
            "The paired t-test indicates "
            "no statistically significant "
            "difference between Standard CSP "
            "and FBCSP."
        )

    if (
        not np.isnan(wilcoxon_p)
        and wilcoxon_p < 0.05
    ):

        wilcoxon_result = (
            "The Wilcoxon signed-rank test "
            "also indicates a statistically "
            "significant difference."
        )

    else:

        wilcoxon_result = (
            "The Wilcoxon signed-rank test "
            "also indicates no statistically "
            "significant difference."
        )

    if np.isnan(cohen_d):

        effect_text = (
            "The effect size could not be determined."
        )

    elif abs(cohen_d) < 0.2:

        effect_text = (
            f"Cohen's d = {cohen_d:.4f}, "
            "indicating a negligible effect."
        )

    elif abs(cohen_d) < 0.5:

        effect_text = (
            f"Cohen's d = {cohen_d:.4f}, "
            "indicating a small effect."
        )

    elif abs(cohen_d) < 0.8:

        effect_text = (
            f"Cohen's d = {cohen_d:.4f}, "
            "indicating a medium effect."
        )

    else:

        effect_text = (
            f"Cohen's d = {cohen_d:.4f}, "
            "indicating a large effect."
        )

    if mean_change > 0:

        direction = "improved"

    elif mean_change < 0:

        direction = "decreased"

    else:

        direction = "remained unchanged"

    return (
        f"For {classifier}, FBCSP {direction} "
        f"mean LOSO accuracy by "
        f"{mean_change:+.2f} percentage points. "
        f"{t_result} "
        f"{wilcoxon_result} "
        f"{effect_text} "
        f"{int(improved) if not np.isnan(improved) else 'N/A'} "
        f"of the evaluated subjects improved, while "
        f"{int(decreased) if not np.isnan(decreased) else 'N/A'} "
        f"decreased."
    )


# ============================================================
# FIND BEST CLASSIFIER
# ============================================================

def find_best_loso_classifier(statistics):

    best_classifier = None
    best_accuracy = -np.inf

    for classifier, stats in statistics.items():

        accuracy = stats[
            "fbcsp_mean"
        ]

        if (
            not np.isnan(accuracy)
            and accuracy > best_accuracy
        ):

            best_accuracy = accuracy
            best_classifier = classifier

    return best_classifier, best_accuracy


# ============================================================
# GENERATE CONCLUSION
# ============================================================

def generate_conclusion(
    statistics,
    improvements
):

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    best_classifier, best_accuracy = (
        find_best_loso_classifier(
            statistics
        )
    )

    standard_means = [
        stats["standard_mean"]
        for stats in statistics.values()
        if not np.isnan(stats["standard_mean"])
    ]

    fbcsp_means = [
        stats["fbcsp_mean"]
        for stats in statistics.values()
        if not np.isnan(stats["fbcsp_mean"])
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
    # Determine overall statistical outcome
    # --------------------------------------------------------

    significant_t = []

    significant_w = []

    for stats in statistics.values():

        if (
            not np.isnan(stats["t_p"])
        ):

            significant_t.append(
                stats["t_p"] < 0.05
            )

        if (
            not np.isnan(stats["wilcoxon_p"])
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

    text.append(
        "3. SUBJECT-INDEPENDENT RESULTS"
    )

    text.append(
        f"Across the evaluated classifiers, the highest mean "
        f"FBCSP LOSO accuracy was obtained by {best_classifier}, "
        f"with a mean accuracy of {best_accuracy:.2f}%."
    )

    text.append(
        f"Considering the three classifiers together, the mean "
        f"accuracy across classifier-level LOSO results was "
        f"{overall_standard:.2f}% for Standard CSP and "
        f"{overall_fbcsp:.2f}% for FBCSP, corresponding to an "
        f"overall change of {overall_change:+.2f} percentage points."
    )

    text.append("")

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

    for classifier in [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]:

        if classifier not in statistics:

            continue

        stats = statistics[
            classifier
        ]

        text.append(
            statistical_interpretation(
                classifier,
                stats
            )
        )

    text.append("")

    if (
        not any_t_significant
        and not any_w_significant
    ):

        text.append(
            "Overall, none of the evaluated classifiers showed "
            "a statistically significant improvement from Standard "
            "CSP to FBCSP under LOSO evaluation. Therefore, the "
            "results do not provide statistical evidence that the "
            "tested FBCSP configuration consistently improves "
            "subject-independent four-class motor imagery "
            "classification."
        )

    else:

        text.append(
            "The statistical analysis indicates that at least "
            "one classifier exhibited a statistically significant "
            "difference between Standard CSP and FBCSP. This "
            "suggests that the effectiveness of frequency-bank "
            "processing may depend on the classifier and the "
            "specific experimental configuration."
        )

    text.append("")

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
            "FBCSP and Standard CSP produced essentially identical "
            "overall LOSO performance."
        )

    text.append("")

    text.append(
        "7. RESEARCH CONCLUSION"
    )

    text.append(
        "Within the conditions tested in this study, Standard CSP "
        "and the implemented FBCSP configuration provide comparable "
        "subject-independent four-class motor imagery performance. "
        "The statistical analysis does not support the claim that "
        "FBCSP consistently outperforms Standard CSP for unseen "
        "subjects."
    )

    text.append(
        "Consequently, the principal finding of this experiment is "
        "that frequency-bank spatial filtering alone is insufficient "
        "to overcome the inter-subject variability of motor imagery "
        "EEG. Improving subject-independent BCI performance likely "
        "requires additional techniques such as domain adaptation, "
        "transfer learning, subject normalization, Riemannian "
        "geometry-based methods, adaptive spatial filtering, or "
        "deep learning approaches designed for cross-subject EEG."
    )

    text.append("")

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

    return "\n".join(text)


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

        writer = csv.writer(file)

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

        for classifier, stats in statistics.items():

            writer.writerow([
                classifier,
                f"{stats['standard_mean']:.4f}",
                f"{stats['fbcsp_mean']:.4f}",
                f"{stats['mean_change']:.4f}",
                f"{stats['t_p']:.6f}",
                f"{stats['wilcoxon_p']:.6f}",
                f"{stats['cohen_d']:.4f}",
                int(stats["improved"])
                if not np.isnan(stats["improved"])
                else "",
                int(stats["decreased"])
                if not np.isnan(stats["decreased"])
                else ""
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
        "\nConclusion directory:",
        CONCLUSION_DIR
    )

    # --------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------

    check_input_files()

    # --------------------------------------------------------
    # Load statistics
    # --------------------------------------------------------

    statistics = load_statistics()

    # --------------------------------------------------------
    # Load subject improvements
    # --------------------------------------------------------

    improvements = (
        load_subject_improvements()
    )

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
    # Save
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

    save_summary(
        statistics
    )

    # --------------------------------------------------------
    # Print
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