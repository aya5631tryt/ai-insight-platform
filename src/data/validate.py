from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset path
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def load_data() -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""
    return pd.read_csv(DATA_PATH)


def validate_data(df: pd.DataFrame) -> None:
    """Run basic data quality checks."""

    # Check that the dataset is not empty
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Check for duplicate rows
    if df.duplicated().any():
        raise ValueError("Dataset contains duplicate rows.")

    # Required columns
    required_columns = {
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    # Check for missing required columns
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Check target values
    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Unexpected values found in Churn column.")

    # Detect hidden empty values represented by a single space
    hidden_empty_values = (df == " ").sum()

    problematic_columns = hidden_empty_values[
        hidden_empty_values > 0
    ]

    if not problematic_columns.empty:
        print("\nHidden empty values detected:")
        print(problematic_columns)

    print("\nData validation passed.")


def main():
    """Run the data validation process."""
    df = load_data()
    validate_data(df)


if __name__ == "__main__":
    main()