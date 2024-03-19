""" Math/Stat function library for VCP screener """

from typing import List, Tuple

import pandas as pd


def compute_ad_value(in_dict: dict) -> List[float]:
    """Compute the Advance-decline index"""
    ad_val = in_dict["adv"] - in_dict["decl"]
    ad_per = (in_dict["adv"] - in_dict["decl"]) / (in_dict["adv"] + in_dict["decl"])  # Net Breadth

    return [ad_val, ad_per]


def compute_nhnl_value(in_dict: dict) -> float:
    """Compute the New highs-lows index"""
    return in_dict["new_high"] - in_dict["new_low"]


def compute_rs_rating(in_df: pd.DataFrame()) -> Tuple[float, float, float]:
    """Compute three different RS rating.
    The input df need to have 12m (>252 row) of data
    RS 1 and RS2 = 3 & 6 months, reference: http://forums.worden.com/default.aspx?g=posts&t=14007
    RS 3 is my own definition."""
    df = in_df.copy()

    _NUM_TRADE_DAYS = 21

    current_close = df["Adj Close"][-1]
    ytd_close = df["Adj Close"][-2]
    close_3m = df["Adj Close"][-3 * _NUM_TRADE_DAYS]
    close_6m = df["Adj Close"][-6 * _NUM_TRADE_DAYS]
    close_9m = df["Adj Close"][-9 * _NUM_TRADE_DAYS]
    # Most of the time it doesn't really have 250 rows because of missing dates. Used 245 instead.
    close_12m = df["Adj Close"][-245]

    # Compute RS rating of the stock (taken from http://forums.worden.com/default.aspx?g=posts&t=14007)
    # 3 months weighted RS Rating
    rs_rating_1 = (
        ((current_close - close_3m) / close_3m) * 40
        + ((current_close - close_6m) / close_6m) * 20
        + ((current_close - close_9m) / close_9m) * 20
        + ((current_close - close_12m) / close_12m) * 20
    )

    # 6 months weighted RS Rating
    rs_rating_2 = (
        ((current_close - close_6m) / close_6m) * 40
        + ((current_close - close_9m) / close_9m) * 20
        + ((current_close - close_12m) / close_12m) * 20
    )

    # RS rating 3
    wi = 1.0 / (len(df.index) + 3 * _NUM_TRADE_DAYS)
    # Doesn't matter - dj['Adj Close'] or not since it is just a constant
    df["Weighted Ratio"] = wi * ((current_close - df["Adj Close"]) / df["Adj Close"])
    df.iloc[-3 * _NUM_TRADE_DAYS :, 6] = df["Weighted Ratio"].iloc[-3 * _NUM_TRADE_DAYS :] * 2.0
    rs_rating_3 = df["Weighted Ratio"].sum() * 100

    return rs_rating_1, rs_rating_2, rs_rating_3


def get_rs_rank_sorted_df(in_df: pd.DataFrame()) -> pd.DataFrame():
    """Compute RS rank from the RS rating column of the dataframe, then return a sorted dataframe"""
    df = in_df.copy()
    sort_column = "RS Rank 3"

    if df.empty:
        return pd.DataFrame(
            columns=["Stock", "Index", "RS Rating", "RS Rating 2", "RS Rating 3", "RS Rank", "RS Rank 3"]
        )

    df["RS Rank"] = round((df.loc[:, "RS Rating"].rank(pct=True) * 1), 5)
    df["RS Rank 3"] = round((df.loc[:, "RS Rating 3"].rank(pct=True) * 1), 5)
    df.sort_values(by=[sort_column], axis=0, inplace=True, ascending=False)

    return df
