from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Processed dataset
PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)


def load_data() -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset."""

    cleaned_df = df.copy()

    # Convert hidden empty strings to missing values
    cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].replace(
        " ",
        pd.NA,
    )

    # Convert TotalCharges from text to numeric
    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"],
        errors="coerce",
    )

    # The missing TotalCharges values belong to customers
    # with zero tenure, so their total charges are 0.
    missing_total_charges = cleaned_df["TotalCharges"].isna()

    if missing_total_charges.any():
        invalid_rows = cleaned_df.loc[
            missing_total_charges & (cleaned_df["tenure"] != 0)
        ]

        if not invalid_rows.empty:
            raise ValueError(
                "Found missing TotalCharges for customers "
                "with non-zero tenure."
            )

        cleaned_df.loc[missing_total_charges, "TotalCharges"] = 0.0

    # Convert target variable to binary values
    cleaned_df["Churn"] = cleaned_df["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    # Remove customer ID because it is an identifier,
    # not a predictive feature.
    cleaned_df = cleaned_df.drop(columns=["customerID"])

    return cleaned_df


def save_data(df: pd.DataFrame) -> None:
    """Save the cleaned dataset."""
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )


def main():
    """Run the complete cleaning process."""

    print("Loading raw dataset...")

    df = load_data()

    print(f"Original shape: {df.shape}")

    cleaned_df = clean_data(df)

    print(f"Cleaned shape: {cleaned_df.shape}")

    print("\n=== Cleaned Data Types ===")
    print(cleaned_df.dtypes)

    print("\n=== Remaining Missing Values ===")
    print(cleaned_df.isna().sum())

    print("\n=== Churn Distribution ===")
    print(cleaned_df["Churn"].value_counts())

    save_data(cleaned_df)

    print("\nCleaned dataset saved successfully.")
    print(f"Path: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()