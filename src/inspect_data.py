from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw dataset path
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Duplicate Rows ===")
    print(df.duplicated().sum())

    print("\n=== Churn Distribution ===")
    print(df["Churn"].value_counts())

    print("\n=== Churn Percentage ===")
    print(df["Churn"].value_counts(normalize=True) * 100)

    print("\n=== Hidden Empty Values ===")
    print((df == " ").sum())

    print("\n=== TotalCharges Unique Problematic Values ===")
    print(df.loc[df["TotalCharges"] == " ", "TotalCharges"])

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()