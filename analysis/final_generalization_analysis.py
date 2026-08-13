import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

PAPER_TABLES_DIR = (
    RESULTS_DIR
    / "paper_tables"
)

FINAL_ANALYSIS_DIR = (
    RESULTS_DIR
    / "final_analysis"
)

FINAL_ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUBJECTS = [
    f"A{i:02d}"
    for i in range(1, 10)
]

CLASSIFIERS = [
    "LDA",
    "SVM",
    "Logistic Regression"
]


# ============================================================
# FILES
# ============================================================

WITHIN_STANDARD_FILE = (
    RESULTS_DIR
    / "four_class_standard_csp_results.csv"
)

WITHIN_FBCSP_FILE = (
    RESULTS_DIR
    / "four_class_fbcsp_results.csv"
)

LOSO_STANDARD_FILE = (
    CROSS_SUBJECT_DIR
    / "standard_csp_loso_results.csv"
)

LOSO_FBCSP_FILE = (
    CROSS_SUBJECT_DIR
    / "fbcsp_loso_results.csv"
)

GENERALIZATION_CSV = (
    FINAL_ANALYSIS_DIR
    / "generalization_gap.csv"
)

GENERALIZATION_PLOT = (
    FINAL_ANALYSIS_DIR
    / "generalization_gap.png"
)

SUMMARY_CSV = (
    FINAL_ANALYSIS_DIR
    / "final_method_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        return list(
            csv.DictReader(file)
        )


def get_float(row, column):

    return float(
        row[column]
    )


# ============================================================
# LOAD WITHIN-SUBJECT RESULTS
# ============================================================

def load_within_subject_results():

    standard_rows = load_csv(
        WITHIN_STANDARD_FILE
    )

    fbcsp_rows = load_csv(
        WITHIN_FBCSP_FILE
    )

    standard = {}
    fbcsp = {}

    # --------------------------------------------------------
    # Standard CSP
    # --------------------------------------------------------

    for classifier in CLASSIFIERS:

        values = []

        for row in standard_rows:

            if row["Classifier"] == classifier:

                values.append(
                    get_float(
                        row,
                        "Mean Accuracy (%)"
                    )
                )

        if len(values) == 0:

            raise ValueError(
                f"No within-subject Standard CSP "
                f"results found for {classifier}"
            )

        standard[classifier] = np.mean(
            values
        )

    # --------------------------------------------------------
    # FBCSP
    # --------------------------------------------------------

    for classifier in CLASSIFIERS:

        values = []

        for row in fbcsp_rows:

            if row["Classifier"] == classifier:

                values.append(
                    get_float(
                        row,
                        "Mean Accuracy (%)"
                    )
                )

        if len(values) == 0:

            raise ValueError(
                f"No within-subject FBCSP "
                f"results found for {classifier}"
            )

        fbcsp[classifier] = np.mean(
            values
        )

    return standard, fbcsp


# ============================================================
# LOAD LOSO RESULTS
# ============================================================

def load_loso_results():

    standard_rows = load_csv(
        LOSO_STANDARD_FILE
    )

    fbcsp_rows = load_csv(
        LOSO_FBCSP_FILE
    )

    standard = {}
    fbcsp = {}

    for classifier in CLASSIFIERS:

        standard_values = []
        fbcsp_values = []

        for row in standard_rows:

            if row["Classifier"] == classifier:

                standard_values.append(
                    get_float(
                        row,
                        "Accuracy (%)"
                    )
                )

        for row in fbcsp_rows:

            if row["Classifier"] == classifier:

                fbcsp_values.append(
                    get_float(
                        row,
                        "Accuracy (%)"
                    )
                )

        if len(standard_values) != 9:

            raise ValueError(
                f"Expected 9 LOSO Standard CSP "
                f"results for {classifier}, "
                f"found {len(standard_values)}"
            )

        if len(fbcsp_values) != 9:

            raise ValueError(
                f"Expected 9 LOSO FBCSP "
                f"results for {classifier}, "
                f"found {len(fbcsp_values)}"
            )

        standard[classifier] = np.mean(
            standard_values
        )

        fbcsp[classifier] = np.mean(
            fbcsp_values
        )

    return standard, fbcsp


# ============================================================
# GENERALIZATION ANALYSIS
# ============================================================

def calculate_analysis():

    within_standard, within_fbcsp = (
        load_within_subject_results()
    )

    loso_standard, loso_fbcsp = (
        load_loso_results()
    )

    rows = []

    for classifier in CLASSIFIERS:

        # ----------------------------------------------------
        # Standard CSP
        # ----------------------------------------------------

        within = within_standard[
            classifier
        ]

        loso = loso_standard[
            classifier
        ]

        gap = within - loso

        retention = (
            loso / within * 100
        )

        rows.append({
            "Method": "Standard CSP",
            "Classifier": classifier,
            "Within-Subject Mean (%)": within,
            "LOSO Mean (%)": loso,
            "Generalization Gap (pp)": gap,
            "LOSO Retention (%)": retention
        })

        # ----------------------------------------------------
        # FBCSP
        # ----------------------------------------------------

        within = within_fbcsp[
            classifier
        ]

        loso = loso_fbcsp[
            classifier
        ]

        gap = within - loso

        retention = (
            loso / within * 100
        )

        rows.append({
            "Method": "FBCSP",
            "Classifier": classifier,
            "Within-Subject Mean (%)": within,
            "LOSO Mean (%)": loso,
            "Generalization Gap (pp)": gap,
            "LOSO Retention (%)": retention
        })

    return rows


# ============================================================
# SAVE GENERALIZATION CSV
# ============================================================

def save_generalization_csv(rows):

    with open(
        GENERALIZATION_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Method",
                "Classifier",
                "Within-Subject Mean (%)",
                "LOSO Mean (%)",
                "Generalization Gap (pp)",
                "LOSO Retention (%)"
            ]
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                "Method":
                    row["Method"],

                "Classifier":
                    row["Classifier"],

                "Within-Subject Mean (%)":
                    f'{row["Within-Subject Mean (%)"]:.4f}',

                "LOSO Mean (%)":
                    f'{row["LOSO Mean (%)"]:.4f}',

                "Generalization Gap (pp)":
                    f'{row["Generalization Gap (pp)"]:.4f}',

                "LOSO Retention (%)":
                    f'{row["LOSO Retention (%)"]:.4f}'
            })


# ============================================================
# PLOT
# ============================================================

def plot_generalization(rows):

    x = np.arange(
        len(CLASSIFIERS)
    )

    width = 0.35

    standard_loso = [
        row["LOSO Mean (%)"]
        for row in rows
        if row["Method"] == "Standard CSP"
    ]

    fbcsp_loso = [
        row["LOSO Mean (%)"]
        for row in rows
        if row["Method"] == "FBCSP"
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars1 = ax.bar(
        x - width / 2,
        standard_loso,
        width,
        label="Standard CSP"
    )

    bars2 = ax.bar(
        x + width / 2,
        fbcsp_loso,
        width,
        label="FBCSP"
    )

    for bar, value in zip(
        bars1,
        standard_loso
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
        fbcsp_loso
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center"
        )

    ax.set_title(
        "Subject-Independent LOSO Performance"
    )

    ax.set_xlabel(
        "Classifier"
    )

    ax.set_ylabel(
        "Mean LOSO Accuracy (%)"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        CLASSIFIERS
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        GENERALIZATION_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SAVE METHOD SUMMARY
# ============================================================

def save_method_summary(rows):

    summary = []

    for classifier in CLASSIFIERS:

        standard = next(
            row for row in rows
            if row["Method"] == "Standard CSP"
            and row["Classifier"] == classifier
        )

        fbcsp = next(
            row for row in rows
            if row["Method"] == "FBCSP"
            and row["Classifier"] == classifier
        )

        loso_change = (
            fbcsp["LOSO Mean (%)"]
            -
            standard["LOSO Mean (%)"]
        )

        within_change = (
            fbcsp["Within-Subject Mean (%)"]
            -
            standard["Within-Subject Mean (%)"]
        )

        summary.append({
            "Classifier": classifier,
            "Within CSP (%)":
                standard["Within-Subject Mean (%)"],
            "Within FBCSP (%)":
                fbcsp["Within-Subject Mean (%)"],
            "Within FBCSP Change (pp)":
                within_change,
            "LOSO CSP (%)":
                standard["LOSO Mean (%)"],
            "LOSO FBCSP (%)":
                fbcsp["LOSO Mean (%)"],
            "LOSO FBCSP Change (pp)":
                loso_change
        })

    with open(
        SUMMARY_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Classifier",
                "Within CSP (%)",
                "Within FBCSP (%)",
                "Within FBCSP Change (pp)",
                "LOSO CSP (%)",
                "LOSO FBCSP (%)",
                "LOSO FBCSP Change (pp)"
            ]
        )

        writer.writeheader()

        for row in summary:

            writer.writerow({
                key:
                    (
                        f"{value:.4f}"
                        if isinstance(value, float)
                        else value
                    )
                for key, value in row.items()
            })


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(rows):

    print("\n")
    print("=" * 110)
    print("FINAL GENERALIZATION ANALYSIS")
    print("=" * 110)

    print(
        f"{'Method':<18}"
        f"{'Classifier':<23}"
        f"{'Within':>12}"
        f"{'LOSO':>12}"
        f"{'Gap':>12}"
        f"{'Retention':>14}"
    )

    print("-" * 110)

    for row in rows:

        print(
            f"{row['Method']:<18}"
            f"{row['Classifier']:<23}"
            f"{row['Within-Subject Mean (%)']:>10.2f}%"
            f"{row['LOSO Mean (%)']:>10.2f}%"
            f"{row['Generalization Gap (pp)']:>10.2f}"
            f"{row['LOSO Retention (%)']:>12.2f}%"
        )

    print("-" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 100)
    print("EIDOS-BCI")
    print("=" * 100)
    print("FINAL GENERALIZATION-GAP ANALYSIS")

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "Output directory:",
        FINAL_ANALYSIS_DIR
    )

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    required_files = [
        WITHIN_STANDARD_FILE,
        WITHIN_FBCSP_FILE,
        LOSO_STANDARD_FILE,
        LOSO_FBCSP_FILE
    ]

    print("\nChecking input files:")

    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(
                f"\nMissing required file:\n{file}"
            )

        print(
            f"✓ {file}"
        )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    rows = calculate_analysis()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_generalization_csv(
        rows
    )

    plot_generalization(
        rows
    )

    save_method_summary(
        rows
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        rows
    )

    print("\nGenerated files:")

    print(
        f"1. {GENERALIZATION_CSV}"
    )

    print(
        f"2. {GENERALIZATION_PLOT}"
    )

    print(
        f"3. {SUMMARY_CSV}"
    )

    print("\n✓ FINAL GENERALIZATION ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()