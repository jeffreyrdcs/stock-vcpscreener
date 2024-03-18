""" Database library for VCP screener """

import os
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

from stock_vcpscreener.vcp_util.util import convert_date_str_to_datetime, get_last_trade_day

pd.options.mode.chained_assignment = None


_GSPC_SP500_CSV_FILENAME = "GSPC_SP500.csv"
_LAST_UPDATE_DAT_FILENAME = "last_update.dat"


def volume_check():
    """Yfinance often gives us wrong volume that is 100x its actual value.
    Need to work out a way to cross-match the two libraries
    """
    pass


def create_index_database(index_dir_name, source, out_filename=_GSPC_SP500_CSV_FILENAME):
    """Fetch S&P500 index and save result as a csv. _GSPC_SP500_csv_filename is the default filename."""
    trade_day = get_last_trade_day()

    index = _get_sp500_index_ticker(source)

    start_date = trade_day.date() - timedelta(days=365 + 7)
    end_date = trade_day.date() + timedelta(days=1)  # +1 to include the current day

    if not os.path.exists(index_dir_name + out_filename):
        print(f"Fetching index data to {out_filename} ...")
        try:
            if source == "yfinance":
                yf.pdr_override()
                df = pdr.get_data_yahoo(index, start=start_date, end=end_date)
            elif source == "stooq":
                df = pdr.DataReader(index, "stooq", start=start_date, end=end_date)
            df.to_csv(index_dir_name + out_filename)

        except Exception as inst:
            print(inst)
    else:
        print(
            f"File {out_filename} exists. Delete the file if you want to re-download the data or update index instead."
        )


def get_index_lastday(index_dir_name, index_filename=_GSPC_SP500_CSV_FILENAME):
    """Return the most updated date of the index csv file"""
    if os.path.exists(index_dir_name + index_filename):
        index_df = pd.read_csv(index_dir_name + index_filename, header=0)
        index_df["Date"] = pd.to_datetime(index_df["Date"])
        index_df.set_index("Date", inplace=True)
        last_avail_date = index_df.index[-1]
    else:
        last_avail_date = None

    return last_avail_date


def update_index_database(index_dir_name, source, trade_day, index_filename=_GSPC_SP500_CSV_FILENAME):
    """Read in csv for the S&P500 index, check the last updated date and update accordingly"""
    index = _get_sp500_index_ticker(source)

    if os.path.exists(index_dir_name + index_filename):
        index_df = pd.read_csv(index_dir_name + index_filename, header=0)
        index_df["Date"] = pd.to_datetime(index_df["Date"])
        index_df.set_index("Date", inplace=True)
        last_avail_date = index_df.index[-1]

        if (trade_day - last_avail_date.date()).days > 0:
            print(f"Updating index data to {index_filename}")
            try:
                if source == "yfinance":
                    yf.pdr_override()
                    df = pdr.get_data_yahoo(
                        index,
                        start=last_avail_date.date() - timedelta(days=15),
                        end=trade_day + timedelta(days=1),
                        pause=1,
                    )  # Get two weeks back
                elif source == "stooq":
                    df = pdr.DataReader(
                        index,
                        "stooq",
                        start=last_avail_date.date() - timedelta(days=15),
                        end=trade_day + timedelta(days=1),
                    )
                index_df = pd.concat([index_df, df])

                # Check if the Volume column of the new data is identical to the one in the db
                check = index_df.groupby(index_df.index)["Volume"].nunique().ne(1)
                if sum(check) != 0:
                    # Now set everything to be the minimum of the duplicated
                    index_df["Volume"] = index_df.groupby(index_df.index)["Volume"].transform("min")

                index_df = index_df[~index_df.index.duplicated(keep="first")]
                index_df = index_df.sort_index()
                index_df.to_csv(index_dir_name + index_filename, mode="w")

                # Determine if the data is updated or not
                updated_last_avail_date = index_df.index[-1]
                if (trade_day - updated_last_avail_date.date()).days > 0:
                    print(f"Last updated date of the index is {updated_last_avail_date}.")
                    print("Please wait until yahoo finance update today's data.")
                    sys.exit(0)

            except Exception as inst:
                print(f"Error updating index database: {inst}")
        else:
            print(f"No update needed")
    else:
        print("Index File not found!")


def _get_sp500_index_ticker(source):
    """Return the ticker for the S&P500 index"""
    if source == "yfinance":
        index = "^GSPC"
    elif source == "stooq":
        index = "^SPX"  # SPX and SPY represent options on the S&P 500 index
    else:
        raise Exception("Please select either yfinance or stooq.")

    return index


def get_stock_filename(stock):
    """Return the filename of the stock data csv"""
    return stock.strip().ljust(5, "_") + ".csv"


def get_stock_data_specific_date(data_dir_name, stock, in_date, minmax_range=False, percent_change=False):
    """Return the OHLC df of a specific stock of a given date"""
    in_stock_filename = get_stock_filename(stock)

    if not isinstance(in_date, date):
        in_date = convert_date_str_to_datetime(in_date)

    if not os.path.exists(data_dir_name + in_stock_filename):
        print(f"Stock {stock} not available")
        return None

    df = pd.read_csv(data_dir_name + in_stock_filename, header=0)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].dt.date
    df.set_index("Date", inplace=True)

    if in_date in df.index:
        sel_df = df.loc[in_date]
        sel_df["Ticker"] = stock.strip()

        if percent_change:
            tmp_df = df[df.index <= in_date]
            sel_df["Change"] = tmp_df["Adj Close"].iloc[-1] - tmp_df["Adj Close"].iloc[-2]
            sel_df["Change (%)"] = (tmp_df["Adj Close"].iloc[-1] - tmp_df["Adj Close"].iloc[-2]) / tmp_df[
                "Adj Close"
            ].iloc[-2]

        if minmax_range:
            sel_df["52 Week Min"] = min(df["Adj Close"].loc[in_date - timedelta(days=365): in_date])
            sel_df["52 Week Max"] = max(df["Adj Close"].loc[in_date - timedelta(days=365): in_date])

        # sel_df = sel_df.drop("Date")
    else:
        print(f"Specified date not available in the data for stock {stock}")
        return None

    return sel_df


def _get_last_update_date_df(data_dir_name, trade_day) -> pd.DataFrame:
    # Read in the database update date
    if os.path.exists(data_dir_name + _LAST_UPDATE_DAT_FILENAME):
        last_update = pd.read_csv(data_dir_name + _LAST_UPDATE_DAT_FILENAME, header=0)
        last_update["Date"] = pd.to_datetime(last_update["Date"])
    else:
        # If the last update file doesn't exist, create a last update df with the date of one year ago
        last_update = pd.DataFrame([trade_day.date() - timedelta(days=365 + 7)], columns=["Date"])

    return last_update


def create_stock_database(stock_list, data_dir_name, source):
    """Download stock data and save them as csv"""
    trade_day = get_last_trade_day()

    start_date = trade_day.date() - timedelta(days=365 + 7)  # One year + 7 days
    end_date = trade_day.date() + timedelta(days=1)

    last_update = _get_last_update_date_df(data_dir_name, trade_day)

    for stock in stock_list:
        out_stock_filename = get_stock_filename(stock)

        if os.path.exists(data_dir_name + out_stock_filename):
            print(f"File {out_stock_filename} exists. Skipping ...")
            continue

        print(f"Fetching {stock} info to {out_stock_filename}")
        try:
            if source == "yfinance":
                yf.pdr_override()
                df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
            elif source == "stooq":
                df = pdr.DataReader(stock.strip(), "stooq", start=start_date, end=end_date)

            if df.empty:
                raise ValueError("Empty DataFrame retrieved.")

            df.to_csv(data_dir_name + out_stock_filename)

        except Exception as e:
            error_file_name = data_dir_name + stock.strip().ljust(5, "_") + ".txt"
            with open(error_file_name, "w") as out_file:
                out_file.write(f"{e}\n")
                out_file.write(f"Last try on {end_date}")
            print(f"Error fetching data for {stock}: {e}")

    # When done, update the last update csv file to current time (UTC-5)
    last_update["Date"] = (datetime.utcnow() - timedelta(hours=5)).date()
    last_update.to_csv(data_dir_name + _LAST_UPDATE_DAT_FILENAME, mode="w", index=False)


def update_stock_database(stock_list, data_dir_name, source, trade_day, override=False):
    """Read in csv for each stock, check the last updated date and update accordingly"""
    # trade_day = get_last_trade_day().date()

    # Read in the database update date
    last_update = pd.read_csv(data_dir_name + _LAST_UPDATE_DAT_FILENAME, header=0)
    last_update["Date"] = pd.to_datetime(last_update["Date"])
    last_update_day = last_update["Date"][0]

    if (trade_day - last_update_day.date()).days > 0 or override:
        for stock in stock_list:

            # For testing
            # if stock[0:1] <'S':
            #     continue

            in_stock_filename = get_stock_filename(stock)

            if os.path.exists(data_dir_name + in_stock_filename):
                stock_data = pd.read_csv(data_dir_name + in_stock_filename, header=0)
                stock_data["Date"] = pd.to_datetime(stock_data["Date"])
                stock_data.set_index("Date", inplace=True)

                # Check if the last line contains Nan:
                if np.isnan(stock_data.iloc[-1]["Open"]) or np.isnan(stock_data.iloc[-1]["Volume"]):
                    stock_data = stock_data[:-1]

                last_avail_date = stock_data.index[-1]

                if (trade_day - last_avail_date.date()).days > 0:
                    # print((trade_day.date() - last_avail_date.date()).days)
                    print(f"Updating info on {stock} on {in_stock_filename}")
                    try:
                        if source == "yfinance":
                            yf.pdr_override()
                            df = pdr.get_data_yahoo(
                                stock,
                                start=last_avail_date.date() - timedelta(days=15),
                                end=trade_day + timedelta(days=1),
                            )  #  pause=1  Get two weeks back
                        elif source == "stooq":
                            df = pdr.DataReader(
                                stock.strip(),
                                "stooq",
                                start=last_avail_date.date() - timedelta(days=15),
                                end=trade_day + timedelta(days=1),
                            )
                        stock_data = pd.concat([stock_data, df])

                        # Check if the Volume column of the new data is identical to the one in the db
                        check = stock_data.groupby(stock_data.index)["Volume"].nunique().ne(1)
                        if sum(check) != 0:
                            # Now set everything to be the minimum of the duplicated
                            stock_data["Volume"] = stock_data.groupby(stock_data.index)["Volume"].transform("min")

                        stock_data = stock_data[~stock_data.index.duplicated(keep="first")]
                        stock_data = stock_data.sort_index()
                        stock_data.to_csv(data_dir_name + in_stock_filename, mode="w")

                    except Exception as e:
                        print(f"Error updating data for {stock}: {e}")
                else:
                    print(f"No update needed for {stock}")

                # Wait a while to avoid data error which seems to be happening a lot for yfinance
                time.sleep(0.4)

        # When done, update the last update file to current time (UTC-5). Now using trade_day instead
        last_update["Date"] = trade_day  # (datetime.utcnow() - timedelta(hours=5)).date()
        last_update.to_csv(data_dir_name + _LAST_UPDATE_DAT_FILENAME, mode="w", index=False)
    else:
        print("No update needed for the database!")
