"""
Feature engineering for the Home Credit Default Risk pipeline.

Each function loads one auxiliary table, encodes categoricals, aggregates
to one row per SK_ID_CURR, and returns the result ready to merge into the
main application table.
"""

import pandas as pd


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns in-place and return the DataFrame."""
    for col in df.select_dtypes("object").columns:
        df[col] = pd.Categorical(df[col]).codes
    return df


def _aggregate(df: pd.DataFrame, group_col: str, prefix: str,
               drop_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Group df by group_col, aggregate with mean/max/min/std/count,
    flatten column names with prefix, and reset the index.

    Parameters
    ----------
    df        : DataFrame already encoded (no object columns).
    group_col : Column to group by (kept as-is after reset_index).
    prefix    : String prepended to every aggregated column name.
    drop_cols : Extra columns to drop before aggregating (e.g. surrogate keys).
    """
    if drop_cols:
        df = df.drop(columns=drop_cols)
    agg = df.groupby(group_col).agg(["mean", "max", "min", "std", "count"])
    agg.columns = [f"{prefix}_{a}_{b}".upper() for a, b in agg.columns]
    return agg.reset_index()



def build_bureau_features(data_dir: str) -> pd.DataFrame:
    """
    Load bureau.csv, encode categoricals, aggregate to SK_ID_CURR level.
    Column names are prefixed with BUREAU_.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    bureau = pd.read_csv(f"{data_dir}/bureau.csv")
    bureau = _encode_categoricals(bureau)
    return _aggregate(bureau, group_col="SK_ID_CURR", prefix="BUREAU",
                      drop_cols=["SK_ID_BUREAU"])


def build_bureau_balance_features(data_dir: str) -> pd.DataFrame:
    """
    Load bureau.csv and bureau_balance.csv. Aggregate bureau_balance to
    SK_ID_BUREAU level, merge into bureau, then aggregate to SK_ID_CURR level.
    Column names are prefixed with BUREAU_BB_.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    bureau = pd.read_csv(f"{data_dir}/bureau.csv")
    bureau_balance = pd.read_csv(f"{data_dir}/bureau_balance.csv")

    bureau_balance = _encode_categoricals(bureau_balance)
    bb_agg = _aggregate(bureau_balance, group_col="SK_ID_BUREAU", prefix="BB")

    bureau = _encode_categoricals(bureau)
    bureau_full = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    return _aggregate(bureau_full, group_col="SK_ID_CURR", prefix="BUREAU_BB",
                      drop_cols=["SK_ID_BUREAU"])


def build_previous_application_features(data_dir: str) -> pd.DataFrame:
    """
    Load previous_application.csv, encode categoricals, aggregate to
    SK_ID_CURR level.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    prev = pd.read_csv(f"{data_dir}/previous_application.csv")
    prev = _encode_categoricals(prev)
    return _aggregate(prev, group_col="SK_ID_CURR", prefix="PREV",
                      drop_cols=["SK_ID_PREV"])


def build_pos_cash_features(data_dir: str) -> pd.DataFrame:
    """
    Load POS_CASH_balance.csv, encode categoricals, aggregate to
    SK_ID_CURR level.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    pos = pd.read_csv(f"{data_dir}/POS_CASH_balance.csv")
    pos = _encode_categoricals(pos)
    return _aggregate(pos, group_col="SK_ID_CURR", prefix="POS",
                      drop_cols=["SK_ID_PREV"])


def build_credit_card_features(data_dir: str) -> pd.DataFrame:
    """
    Load credit_card_balance.csv, encode categoricals, aggregate to
    SK_ID_CURR level.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    cc = pd.read_csv(f"{data_dir}/credit_card_balance.csv")
    cc = _encode_categoricals(cc)
    return _aggregate(cc, group_col="SK_ID_CURR", prefix="CC",
                      drop_cols=["SK_ID_PREV"])


def build_installments_features(data_dir: str) -> pd.DataFrame:
    """
    Load installments_payments.csv, encode categoricals, aggregate to
    SK_ID_CURR level.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    inst = pd.read_csv(f"{data_dir}/installments_payments.csv")
    inst = _encode_categoricals(inst)
    return _aggregate(inst, group_col="SK_ID_CURR", prefix="INST",
                      drop_cols=["SK_ID_PREV"])


def build_features(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """
    Join all auxiliary table aggregations onto the main application DataFrame.

    Parameters
    ----------
    df       : application_train.csv or application_test.csv already loaded.
    data_dir : Path to the directory containing all raw CSV files.

    Returns the enriched DataFrame with one row per applicant.
    """
    builders = [
        build_bureau_features,
        build_bureau_balance_features,
        build_previous_application_features,
        build_pos_cash_features,
        build_credit_card_features,
        build_installments_features,
    ]

    for build_fn in builders:
        agg = build_fn(data_dir)
        df = df.merge(agg, on="SK_ID_CURR", how="left")
        print(f"  [{build_fn.__name__}] shape after merge: {df.shape}")

    # Encode any remaining categoricals in the main table
    df = _encode_categoricals(df)
    return df