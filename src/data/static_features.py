"""Static (loan origination) feature loading, cleaning, and merging.

Ported from the notebook cells that clean borrower/property/loan static
features and merge them onto the time-series sequence table by LOAN_NUMBER.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import STATIC_CATEGORICAL_COLS, STATIC_NUMERIC_COLS


def load_static_features(csv_path: str) -> pd.DataFrame:
    static_df = pd.read_csv(csv_path, dtype=str)
    static_df["ZIP3"] = static_df["ZIP3"].astype(str).str[:-2]
    static_df[STATIC_NUMERIC_COLS] = static_df[STATIC_NUMERIC_COLS].astype(float)
    static_df["NUMBER_OF_BORROWERS"] = static_df["NUMBER_OF_BORROWERS"].astype(int)
    return static_df


def prepare_static_features(
    static_df: pd.DataFrame,
    loan_numbers_train: pd.Series,
    loan_numbers_test: pd.Series,
    numeric_cols=STATIC_NUMERIC_COLS,
    categorical_cols=STATIC_CATEGORICAL_COLS,
):
    """Scale numeric columns (fit on train) and one-hot encode categorical
    columns (fit on train), returning one finished DataFrame per split with
    LOAN_NUMBER/NUMBER_OF_BORROWERS/MSA/ZIP3 preserved unscaled for merging.
    """
    static_train_val = static_df[static_df["LOAN_NUMBER"].isin(loan_numbers_train)].copy()
    static_test = static_df[static_df["LOAN_NUMBER"].isin(loan_numbers_test)].copy()

    scaler = StandardScaler()
    static_train_val[numeric_cols] = scaler.fit_transform(static_train_val[numeric_cols])
    static_test[numeric_cols] = scaler.transform(static_test[numeric_cols])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_train = encoder.fit_transform(static_train_val[categorical_cols])
    encoded_test = encoder.transform(static_test[categorical_cols])

    encoded_train_df = pd.DataFrame(
        encoded_train, columns=encoder.get_feature_names_out(categorical_cols), index=static_train_val.index
    )
    encoded_test_df = pd.DataFrame(
        encoded_test, columns=encoder.get_feature_names_out(categorical_cols), index=static_test.index
    )

    keep_cols = ["LOAN_NUMBER", "NUMBER_OF_BORROWERS", "MSA", "ZIP3"]
    static_train_final = pd.concat(
        [static_train_val[keep_cols], static_train_val[numeric_cols], encoded_train_df], axis=1
    )
    static_test_final = pd.concat([static_test[keep_cols], static_test[numeric_cols], encoded_test_df], axis=1)

    return static_train_final, static_test_final


def merge_time_series_with_static(
    x_train_val: pd.DataFrame,
    x_test: pd.DataFrame,
    loan_numbers_train: pd.Series,
    loan_numbers_test: pd.Series,
    static_train_final: pd.DataFrame,
    static_test_final: pd.DataFrame,
):
    """Attach LOAN_NUMBER to each time-series row, merge in the static
    features, then drop LOAN_NUMBER again (kept only for the join).
    """
    x_train_val = x_train_val.copy()
    x_test = x_test.copy()

    x_train_val["LOAN_NUMBER"] = loan_numbers_train.values
    x_test["LOAN_NUMBER"] = loan_numbers_test.values

    x_train_val = x_train_val.merge(static_train_final, on="LOAN_NUMBER", how="left")
    x_test = x_test.merge(static_test_final, on="LOAN_NUMBER", how="left")

    x_train_val.drop(columns=["LOAN_NUMBER"], inplace=True)
    x_test.drop(columns=["LOAN_NUMBER"], inplace=True)

    return x_train_val, x_test
