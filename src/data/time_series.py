"""Loan performance time-series loading, delinquency labeling, and scaling.

Ported from the original notebook cells that build monthly loan sequences
from the Freddie Mac performance extract and label each loan delinquent if
it was ever delinquent during the prediction window.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import LABEL_WINDOW_END, LABEL_WINDOW_START, SEED, TEST_SIZE, TIME_SERIES_COLS


def load_loan_performance(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def label_delinquency(df: pd.DataFrame) -> pd.DataFrame:
    """Assign target=1 to loans delinquent at least once in the label window,
    then drop the label-window rows so the model only sees prior history.
    """
    df = df.copy()
    df["REPORTING_PERIOD"] = pd.to_datetime(df["REPORTING_PERIOD"], format="%Y%m")
    df["CURRENT_LOAN_DELINQUENCY_STATUS"] = pd.to_numeric(
        df["CURRENT_LOAN_DELINQUENCY_STATUS"], errors="coerce"
    ).fillna(1)

    df = df.sort_values(by=["LOAN_NUMBER", "REPORTING_PERIOD"])
    df["CURRENT_LOAN_DELINQUENCY_STATUS"] = (df["CURRENT_LOAN_DELINQUENCY_STATUS"] > 0).astype(int)

    mask = (df["REPORTING_PERIOD"] >= LABEL_WINDOW_START) & (df["REPORTING_PERIOD"] <= LABEL_WINDOW_END)
    df["target"] = (
        df.groupby("LOAN_NUMBER")["CURRENT_LOAN_DELINQUENCY_STATUS"].transform(lambda x: x[mask].max()).astype(int)
    )

    # Original lab drops a slightly wider window (Mar 31 - Jun 2) than the
    # labeling window above (Apr 1 - Jun 30); kept as-is to match the source.
    df = df[~df["REPORTING_PERIOD"].between(pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-02"))]
    return df


def build_loan_sequences(df: pd.DataFrame, num_cols=TIME_SERIES_COLS, seed=SEED, test_size=TEST_SIZE) -> pd.DataFrame:
    """Group monthly rows into one fixed-length sequence per loan, and mark a
    stratified train/test split. Returns a DataFrame indexed by LOAN_NUMBER
    with one array-valued column per feature in `num_cols`, plus
    'target' and 'if_test' columns.
    """
    grouped = df.groupby("LOAN_NUMBER")
    x_dict, y_dict = {}, {}
    for loan_id, group in grouped:
        group = group.sort_values("REPORTING_PERIOD")
        x_dict[loan_id] = {col: group[col].values for col in num_cols}
        y_dict[loan_id] = group["target"].iloc[0]

    np.random.seed(seed)
    x_df = pd.DataFrame.from_dict(x_dict, orient="index")
    x_df["target"] = x_df.index.map(y_dict)

    train_idx, test_idx = train_test_split(
        x_df.index, test_size=test_size, stratify=x_df["target"], random_state=seed
    )
    x_df["if_test"] = 0
    x_df.loc[test_idx, "if_test"] = 1
    return x_df


def scale_dataframe_sequences(x_df: pd.DataFrame, num_cols=TIME_SERIES_COLS, test_var=None):
    """Standardize each time-series feature independently (fit on train only),
    leaving the binary delinquency-status feature unscaled.
    """
    scaler = StandardScaler()

    if test_var is not None:
        x_train = x_df.loc[test_var == 0, :]
        x_test = x_df.loc[test_var == 1, :]
    else:
        x_train = x_df
        x_test = None

    scaled_data, scaled_data_test = {}, {}
    seq_len = len(next(iter(x_train.iloc[0])))

    for column in num_cols:
        data = np.stack(x_train[column].values).reshape(-1, seq_len)
        if column != "CURRENT_LOAN_DELINQUENCY_STATUS":
            scaled_data[column] = scaler.fit_transform(data).reshape(-1, 1, seq_len).tolist()
        else:
            scaled_data[column] = data.reshape(-1, 1, seq_len).tolist()

        if x_test is not None:
            data_test = np.stack(x_test[column].values).reshape(-1, seq_len)
            if column != "CURRENT_LOAN_DELINQUENCY_STATUS":
                scaled_data_test[column] = scaler.transform(data_test).reshape(-1, 1, seq_len).tolist()
            else:
                scaled_data_test[column] = data_test.reshape(-1, 1, seq_len).tolist()

    scaled_df = pd.DataFrame(scaled_data).map(lambda x: np.array(x[0]))
    if x_test is not None:
        scaled_df_test = pd.DataFrame(scaled_data_test).map(lambda x: np.array(x[0]))
        return scaled_df, scaled_df_test
    return scaled_df, None


@dataclass
class TimeSeriesData:
    x_train_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train_val: np.ndarray
    y_test: np.ndarray
    pos_weight: torch.Tensor
    loan_numbers_train: pd.Series
    loan_numbers_test: pd.Series


def prepare_time_series(csv_path: str, num_cols=TIME_SERIES_COLS, seed=SEED, test_size=TEST_SIZE) -> TimeSeriesData:
    """End-to-end time-series preparation: load -> label -> sequence -> scale."""
    df = load_loan_performance(csv_path)
    df = label_delinquency(df)
    x_df = build_loan_sequences(df, num_cols=num_cols, seed=seed, test_size=test_size)

    x_train_val, x_test = scale_dataframe_sequences(x_df[num_cols], num_cols=num_cols, test_var=x_df["if_test"])

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    y_train_val = enc.fit_transform(x_df.loc[x_df["if_test"] == 0, "target"].values.reshape(-1, 1))[:, 1]
    y_test = enc.transform(x_df.loc[x_df["if_test"] == 1, "target"].values.reshape(-1, 1))[:, 1]

    pos_weight = torch.tensor(np.sum(1 - y_train_val) / np.sum(y_train_val), dtype=torch.float32)

    x_df["LOAN_NUMBER"] = x_df.index
    loan_numbers_train = x_df.loc[x_df["if_test"] == 0, "LOAN_NUMBER"]
    loan_numbers_test = x_df.loc[x_df["if_test"] == 1, "LOAN_NUMBER"]

    return TimeSeriesData(
        x_train_val=x_train_val,
        x_test=x_test,
        y_train_val=y_train_val,
        y_test=y_test,
        pos_weight=pos_weight,
        loan_numbers_train=loan_numbers_train,
        loan_numbers_test=loan_numbers_test,
    )
