from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "four_class"
)

CROSS_SUBJECT_DIR = (
    RESULTS_DIR
    / "cross_subject"
)

STATISTICS_DIR = (
    CROSS_SUBJECT_DIR
    / "statistical_tests"
)

PAPER_TABLES_DIR = (
    RESULTS_DIR
    / "paper_tables"
)

PAPER_TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

BASELINE_FILE = (
    RESULTS_DIR
    / "four_class_standard_csp_results.csv"
)

FBCSP_FILE = (
    RESULTS_DIR
    / "four_class_fbcsp_results.csv"
)

SUBJECT_COMPARISON_FILE = (
    RESULTS_DIR
    / "four_class_subject_results.csv"
)

LOSO_BASELINE_FILE = (
    CROSS_SUBJECT_DIR
    / "standard_csp_loso_results.csv"
)

LOSO_FBCSP_FILE = (
    CROSS_SUBJECT_DIR
    / "fbcsp_loso_results.csv"
)

LOSO_STATISTICS_FILE = (
    STATISTICS_DIR
    / "loso_statistical_comparison.csv"
)

LOSO_IMPROVEMENTS_FILE = (
    STATISTICS_DIR
    / "loso_subject_improvements.csv"
)


# ============================================================
# HELPERS
# ============================================================

def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print(f"[FOUND] {path}")


def save_table(df, filename):

    output_file = (
        PAPER_TABLES_DIR
        / filename
    )

    df.to_csv(
        output_file,
        index=False,
        float_format="%.4f"
    )

    print(
        f"[SAVED] {output_file}"
    )


# ============================================================
# TABLE 1
# WITHIN-SUBJECT STANDARD CSP
# ============================================================

def create_table_1():

    df = pd.read_csv(
        BASELINE_FILE
    )

    table = df.pivot(
        index="Subject",
        columns="Classifier",
        values="Mean Accuracy (%)"
    ).reset_index()

    table = table.rename(
        columns={
            "LDA": "LDA (%)",
            "SVM": "SVM (%)",
            "Logistic Regression":
                "Logistic Regression (%)"
        }
    )

    table = table[
        [
            "Subject",
            "LDA (%)",
            "SVM (%)",
            "Logistic Regression (%)"
        ]
    ]

    save_table(
        table,
        "table_1_within_subject_standard_csp.csv"
    )

    return table


# ============================================================
# TABLE 2
# WITHIN-SUBJECT FBCSP
# ============================================================

def create_table_2():

    df = pd.read_csv(
        FBCSP_FILE
    )

    table = df.pivot(
        index="Subject",
        columns="Classifier",
        values="Mean Accuracy (%)"
    ).reset_index()

    table = table.rename(
        columns={
            "LDA": "LDA (%)",
            "SVM": "SVM (%)",
            "Logistic Regression":
                "Logistic Regression (%)"
        }
    )

    table = table[
        [
            "Subject",
            "LDA (%)",
            "SVM (%)",
            "Logistic Regression (%)"
        ]
    ]

    save_table(
        table,
        "table_2_within_subject_fbcsp.csv"
    )

    return table


# ============================================================
# TABLE 3
# LOSO RESULTS
# ============================================================

def create_loso_table():

    standard = pd.read_csv(
        LOSO_BASELINE_FILE
    )

    fbcsp = pd.read_csv(
        LOSO_FBCSP_FILE
    )

    standard_table = standard.pivot(
        index="Test Subject",
        columns="Classifier",
        values="Accuracy (%)"
    ).reset_index()

    fbcsp_table = fbcsp.pivot(
        index="Test Subject",
        columns="Classifier",
        values="Accuracy (%)"
    ).reset_index()

    standard_table.columns = [
        "Test Subject",
        "LDA CSP (%)",
        "SVM CSP (%)",
        "Logistic CSP (%)"
    ]

    fbcsp_table.columns = [
        "Test Subject",
        "LDA FBCSP (%)",
        "SVM FBCSP (%)",
        "Logistic FBCSP (%)"
    ]

    table = standard_table.merge(
        fbcsp_table,
        on="Test Subject"
    )

    table = table[
        [
            "Test Subject",
            "LDA CSP (%)",
            "LDA FBCSP (%)",
            "SVM CSP (%)",
            "SVM FBCSP (%)",
            "Logistic CSP (%)",
            "Logistic FBCSP (%)"
        ]
    ]

    save_table(
        table,
        "table_3_loso_performance.csv"
    )

    return table


# ============================================================
# TABLE 4
# STATISTICAL COMPARISON
# ============================================================

def create_table_4():

    df = pd.read_csv(
        LOSO_STATISTICS_FILE
    )

    # Print the columns so we can verify the
    # exact names generated by your script.

    print(
        "\nStatistical CSV columns:"
    )

    print(
        list(df.columns)
    )

    # Your statistical script should contain
    # the following information.

    possible_mapping = {}

    for column in df.columns:

        lower = column.lower()

        if "classifier" in lower:
            possible_mapping["classifier"] = column

        elif "csp" in lower and "mean" in lower:
            possible_mapping["csp"] = column

        elif "fbcsp" in lower and "mean" in lower:
            possible_mapping["fbcsp"] = column

        elif "change" in lower:
            possible_mapping["change"] = column

        elif "t" in lower and "p" in lower:
            possible_mapping["t_p"] = column

        elif "wilcoxon" in lower and "p" in lower:
            possible_mapping["wilcoxon_p"] = column

        elif "cohen" in lower:
            possible_mapping["cohen_d"] = column

    print(
        "\nDetected columns:"
    )

    print(
        possible_mapping
    )

    # If your CSV uses exactly these names,
    # this will work directly.

    table = df.copy()

    rename_map = {}

    if "classifier" in possible_mapping:
        rename_map[
            possible_mapping["classifier"]
        ] = "Classifier"

    if "csp" in possible_mapping:
        rename_map[
            possible_mapping["csp"]
        ] = "Standard CSP (%)"

    if "fbcsp" in possible_mapping:
        rename_map[
            possible_mapping["fbcsp"]
        ] = "FBCSP (%)"

    if "change" in possible_mapping:
        rename_map[
            possible_mapping["change"]
        ] = "Change (pp)"

    if "t_p" in possible_mapping:
        rename_map[
            possible_mapping["t_p"]
        ] = "Paired t-test p"

    if "wilcoxon_p" in possible_mapping:
        rename_map[
            possible_mapping["wilcoxon_p"]
        ] = "Wilcoxon p"

    if "cohen_d" in possible_mapping:
        rename_map[
            possible_mapping["cohen_d"]
        ] = "Cohen's d"

    table = table.rename(
        columns=rename_map
    )

    save_table(
        table,
        "table_4_loso_statistics.csv"
    )

    return table


# ============================================================
# TABLE 5
# SUBJECT-LEVEL IMPROVEMENTS
# ============================================================

def create_table_5():

    df = pd.read_csv(
        LOSO_IMPROVEMENTS_FILE
    )

    save_table(
        df,
        "table_5_loso_subject_improvements.csv"
    )

    return df


# ============================================================
# TABLE 6
# OVERALL SUMMARY
# ============================================================

def create_table_6():

    standard = pd.read_csv(
        LOSO_BASELINE_FILE
    )

    fbcsp = pd.read_csv(
        LOSO_FBCSP_FILE
    )

    rows = []

    for classifier in [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]:

        standard_values = standard[
            standard["Classifier"]
            == classifier
        ]["Accuracy (%)"]

        fbcsp_values = fbcsp[
            fbcsp["Classifier"]
            == classifier
        ]["Accuracy (%)"]

        rows.append({

            "Classifier":
                classifier,

            "Standard CSP Mean (%)":
                standard_values.mean(),

            "Standard CSP SD (%)":
                standard_values.std(
                    ddof=1
                ),

            "FBCSP Mean (%)":
                fbcsp_values.mean(),

            "FBCSP SD (%)":
                fbcsp_values.std(
                    ddof=1
                ),

            "Change (pp)":
                (
                    fbcsp_values.mean()
                    -
                    standard_values.mean()
                )
        })

    table = pd.DataFrame(
        rows
    )

    save_table(
        table,
        "table_6_overall_loso_summary.csv"
    )

    return table


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 100)
    print("EIDOS-BCI")
    print("PAPER EXPERIMENT TABLE GENERATOR")
    print("=" * 100)

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "\nPaper tables directory:",
        PAPER_TABLES_DIR
    )

    print(
        "\nChecking required files..."
    )

    files = [
        BASELINE_FILE,
        FBCSP_FILE,
        LOSO_BASELINE_FILE,
        LOSO_FBCSP_FILE,
        LOSO_STATISTICS_FILE,
        LOSO_IMPROVEMENTS_FILE
    ]

    for file in files:
        check_file(file)

    print("\n")
    print("=" * 100)
    print("GENERATING PAPER TABLES")
    print("=" * 100)

    create_table_1()

    create_table_2()

    create_loso_table()

    create_table_4()

    create_table_5()

    create_table_6()

    print("\n")
    print("=" * 100)
    print("PAPER TABLE GENERATION COMPLETE")
    print("=" * 100)

    print(
        "\nTables saved in:"
    )

    print(
        PAPER_TABLES_DIR
    )


if __name__ == "__main__":
    main()