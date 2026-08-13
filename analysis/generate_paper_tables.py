from pathlib import Path
import pandas as pd
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

WITHIN_STANDARD_FILE = (
    RESULTS_DIR
    / "four_class_standard_csp_results.csv"
)

WITHIN_FBCSP_FILE = (
    RESULTS_DIR
    / "four_class_fbcsp_results.csv"
)

WITHIN_COMPARISON_FILE = (
    RESULTS_DIR
    / "four_class_comparison.csv"
)

WITHIN_SUBJECT_FILE = (
    RESULTS_DIR
    / "four_class_subject_results.csv"
)

LOSO_STANDARD_FILE = (
    CROSS_SUBJECT_DIR
    / "standard_csp_loso_results.csv"
)

LOSO_FBCSP_FILE = (
    CROSS_SUBJECT_DIR
    / "fbcsp_loso_results.csv"
)

LOSO_COMPARISON_FILE = (
    CROSS_SUBJECT_DIR
    / "loso_comparison.csv"
)

LOSO_SUBJECT_FILE = (
    CROSS_SUBJECT_DIR
    / "subject_results.csv"
)

STATISTICAL_FILE = (
    STATISTICS_DIR
    / "loso_statistical_comparison.csv"
)

IMPROVEMENT_FILE = (
    STATISTICS_DIR
    / "loso_subject_improvements.csv"
)


# ============================================================
# HELPERS
# ============================================================

def section(title):

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)


def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"✓ {path}")


def save_table(df, filename):

    output = (
        PAPER_TABLES_DIR
        / filename
    )

    df.to_csv(
        output,
        index=False,
        float_format="%.4f"
    )

    print(
        f"✓ Saved: {output}"
    )

    return output


# ============================================================
# TABLE 1
# WITHIN-SUBJECT STANDARD CSP
# ============================================================

def create_table_1():

    section(
        "TABLE 1 - WITHIN-SUBJECT STANDARD CSP"
    )

    df = pd.read_csv(
        WITHIN_STANDARD_FILE
    )

    print(
        "Input columns:",
        list(df.columns)
    )

    # --------------------------------------------------------
    # Convert long-format results into a subject × classifier
    # table if necessary.
    # --------------------------------------------------------

    if {
        "Subject",
        "Classifier",
        "Accuracy (%)"
    }.issubset(df.columns):

        table = (
            df
            .pivot(
                index="Subject",
                columns="Classifier",
                values="Accuracy (%)"
            )
            .reset_index()
        )

    elif {
        "Test Subject",
        "Classifier",
        "Accuracy (%)"
    }.issubset(df.columns):

        table = (
            df
            .pivot(
                index="Test Subject",
                columns="Classifier",
                values="Accuracy (%)"
            )
            .reset_index()
        )

    else:

        # If already wide format
        table = df.copy()

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    rename_map = {
        "Test Subject": "Subject",
        "LDA": "LDA Accuracy (%)",
        "SVM": "SVM Accuracy (%)",
        "Logistic Regression":
            "Logistic Regression Accuracy (%)"
    }

    table = table.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # Sort subjects
    # --------------------------------------------------------

    if "Subject" in table.columns:

        table = table.sort_values(
            "Subject"
        )

    return save_table(
        table,
        "table_1_within_subject_standard_csp.csv"
    )


# ============================================================
# TABLE 2
# WITHIN-SUBJECT FBCSP
# ============================================================

def create_table_2():

    section(
        "TABLE 2 - WITHIN-SUBJECT FBCSP"
    )

    df = pd.read_csv(
        WITHIN_FBCSP_FILE
    )

    print(
        "Input columns:",
        list(df.columns)
    )

    if {
        "Subject",
        "Classifier",
        "Accuracy (%)"
    }.issubset(df.columns):

        table = (
            df
            .pivot(
                index="Subject",
                columns="Classifier",
                values="Accuracy (%)"
            )
            .reset_index()
        )

    elif {
        "Test Subject",
        "Classifier",
        "Accuracy (%)"
    }.issubset(df.columns):

        table = (
            df
            .pivot(
                index="Test Subject",
                columns="Classifier",
                values="Accuracy (%)"
            )
            .reset_index()
        )

    else:

        table = df.copy()

    rename_map = {
        "Test Subject": "Subject",
        "LDA": "LDA Accuracy (%)",
        "SVM": "SVM Accuracy (%)",
        "Logistic Regression":
            "Logistic Regression Accuracy (%)"
    }

    table = table.rename(
        columns=rename_map
    )

    if "Subject" in table.columns:

        table = table.sort_values(
            "Subject"
        )

    return save_table(
        table,
        "table_2_within_subject_fbcsp.csv"
    )


# ============================================================
# TABLE 3
# LOSO PERFORMANCE
# ============================================================

def create_table_3():

    section(
        "TABLE 3 - LOSO PERFORMANCE"
    )

    standard = pd.read_csv(
        LOSO_STANDARD_FILE
    )

    fbcsp = pd.read_csv(
        LOSO_FBCSP_FILE
    )

    # --------------------------------------------------------
    # Standard CSP
    # --------------------------------------------------------

    standard_pivot = (
        standard
        .pivot(
            index="Test Subject",
            columns="Classifier",
            values="Accuracy (%)"
        )
    )

    standard_pivot.columns = [
        f"CSP {column} (%)"
        for column in standard_pivot.columns
    ]

    # --------------------------------------------------------
    # FBCSP
    # --------------------------------------------------------

    fbcsp_pivot = (
        fbcsp
        .pivot(
            index="Test Subject",
            columns="Classifier",
            values="Accuracy (%)"
        )
    )

    fbcsp_pivot.columns = [
        f"FBCSP {column} (%)"
        for column in fbcsp_pivot.columns
    ]

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    table = pd.concat(
        [
            standard_pivot,
            fbcsp_pivot
        ],
        axis=1
    ).reset_index()

    table = table.rename(
        columns={
            "Test Subject": "Subject"
        }
    )

    # --------------------------------------------------------
    # Put classifiers in consistent order
    # --------------------------------------------------------

    desired_columns = [
        "Subject",

        "CSP LDA (%)",
        "FBCSP LDA (%)",

        "CSP SVM (%)",
        "FBCSP SVM (%)",

        "CSP Logistic Regression (%)",
        "FBCSP Logistic Regression (%)"
    ]

    existing_columns = [
        column
        for column in desired_columns
        if column in table.columns
    ]

    table = table[
        existing_columns
    ]

    table = table.sort_values(
        "Subject"
    )

    return save_table(
        table,
        "table_3_loso_performance.csv"
    )


# ============================================================
# TABLE 4
# LOSO STATISTICAL RESULTS
# ============================================================

def create_table_4():

    section(
        "TABLE 4 - LOSO STATISTICAL RESULTS"
    )

    df = pd.read_csv(
        STATISTICAL_FILE
    )

    print(
        "Input columns:",
        list(df.columns)
    )

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    rename_map = {

        "Classifier":
            "Classifier",

        "Standard CSP Mean (%)":
            "CSP Mean (%)",

        "FBCSP Mean (%)":
            "FBCSP Mean (%)",

        "Mean Improvement (pp)":
            "Mean Change (pp)",

        "Improvement (pp)":
            "Mean Change (pp)",

        "t-statistic":
            "Paired t-statistic",

        "p-value":
            "Paired t-test p-value",

        "Wilcoxon p-value":
            "Wilcoxon p-value",

        "Cohen's d":
            "Cohen's d"
    }

    table = df.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # Keep useful columns
    # --------------------------------------------------------

    preferred = [
        "Classifier",
        "CSP Mean (%)",
        "FBCSP Mean (%)",
        "Mean Change (pp)",
        "Paired t-statistic",
        "Paired t-test p-value",
        "Wilcoxon p-value",
        "Cohen's d"
    ]

    available = [
        column
        for column in preferred
        if column in table.columns
    ]

    table = table[
        available
    ]

    return save_table(
        table,
        "table_4_loso_statistics.csv"
    )


# ============================================================
# TABLE 5
# SUBJECT-LEVEL IMPROVEMENTS
# ============================================================

def create_table_5():

    section(
        "TABLE 5 - LOSO SUBJECT-LEVEL IMPROVEMENTS"
    )

    df = pd.read_csv(
        IMPROVEMENT_FILE
    )

    print(
        "Input columns:",
        list(df.columns)
    )

    # --------------------------------------------------------
    # If file is already in a useful wide format
    # --------------------------------------------------------

    table = df.copy()

    # --------------------------------------------------------
    # Sort by subject if present
    # --------------------------------------------------------

    subject_column = None

    for candidate in [
        "Subject",
        "Test Subject"
    ]:

        if candidate in table.columns:

            subject_column = candidate
            break

    if subject_column:

        table = table.sort_values(
            subject_column
        )

    return save_table(
        table,
        "table_5_loso_subject_improvements.csv"
    )


# ============================================================
# TABLE 6
# OVERALL LOSO SUMMARY
# ============================================================

def create_table_6():

    section(
        "TABLE 6 - OVERALL LOSO SUMMARY"
    )

    standard = pd.read_csv(
        LOSO_STANDARD_FILE
    )

    fbcsp = pd.read_csv(
        LOSO_FBCSP_FILE
    )

    classifiers = [
        "LDA",
        "SVM",
        "Logistic Regression"
    ]

    rows = []

    for classifier in classifiers:

        standard_values = (
            standard[
                standard["Classifier"]
                == classifier
            ]["Accuracy (%)"]
            .astype(float)
            .to_numpy()
        )

        fbcsp_values = (
            fbcsp[
                fbcsp["Classifier"]
                == classifier
            ]["Accuracy (%)"]
            .astype(float)
            .to_numpy()
        )

        csp_mean = np.mean(
            standard_values
        )

        csp_std = np.std(
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

        change = (
            fbcsp_mean
            - csp_mean
        )

        rows.append({

            "Classifier":
                classifier,

            "CSP Mean (%)":
                csp_mean,

            "CSP SD (%)":
                csp_std,

            "FBCSP Mean (%)":
                fbcsp_mean,

            "FBCSP SD (%)":
                fbcsp_std,

            "Change (pp)":
                change,

        })

    table = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Add statistical information if available
    # --------------------------------------------------------

    if STATISTICAL_FILE.exists():

        stats = pd.read_csv(
            STATISTICAL_FILE
        )

        # Find likely columns
        for classifier in classifiers:

            matching = stats[
                stats["Classifier"]
                == classifier
            ]

            if len(matching) == 0:
                continue

            index = table[
                table["Classifier"]
                == classifier
            ].index[0]

            row = matching.iloc[0]

            # Possible column names
            for source, target in [

                (
                    "Paired t-test p-value",
                    "Paired t-test p-value"
                ),

                (
                    "p-value",
                    "Paired t-test p-value"
                ),

                (
                    "Wilcoxon p-value",
                    "Wilcoxon p-value"
                ),

                (
                    "Cohen's d",
                    "Cohen's d"
                )
            ]:

                if source in row.index:

                    table.loc[
                        index,
                        target
                    ] = row[source]

    return save_table(
        table,
        "table_6_overall_loso_summary.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "EIDOS-BCI PAPER TABLE GENERATION"
    )

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "\nPaper tables directory:",
        PAPER_TABLES_DIR
    )

    # --------------------------------------------------------
    # Verify required files
    # --------------------------------------------------------

    section(
        "CHECKING INPUT FILES"
    )

    required_files = [

        WITHIN_STANDARD_FILE,
        WITHIN_FBCSP_FILE,

        LOSO_STANDARD_FILE,
        LOSO_FBCSP_FILE,

        STATISTICAL_FILE,
        IMPROVEMENT_FILE
    ]

    for file in required_files:

        check_file(file)

    # --------------------------------------------------------
    # Generate tables
    # --------------------------------------------------------

    create_table_1()

    create_table_2()

    create_table_3()

    create_table_4()

    create_table_5()

    create_table_6()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    section(
        "PAPER TABLE GENERATION COMPLETE"
    )

    print(
        "\nGenerated files:"
    )

    for file in sorted(
        PAPER_TABLES_DIR.glob("*.csv")
    ):

        print(
            f"  ✓ {file.name}"
        )

    print(
        "\nLocation:"
    )

    print(
        PAPER_TABLES_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()