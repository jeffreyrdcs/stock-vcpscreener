#!/usr/bin/env python
# from pandas_datareader import data as pdr
import os
import sys
import time
from datetime import date, timedelta

import matplotlib
import mplfinance as mpf
import pandas as pd
import yfinance as yf

from stock_vcpscreener.vcp_util.db import (
    create_index_database,
    create_stock_database,
    get_index_lastday,
    get_stock_data_specific_date,
    get_stock_filename,
    update_index_database,
    update_stock_database,
)
from stock_vcpscreener.vcp_util.stat import compute_rs_rank, compute_rs_rating
from stock_vcpscreener.vcp_util.util import (
    cleanup_dir_jpg_png,
    convert_png_to_jpg,
    convert_report_dict_to_df,
    generate_report_breadth_page,
    generate_combined_pdf_report,
    generate_report_front_page,
    generate_report_output_page,
    get_last_trade_day,
)

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")
matplotlib.use("Agg")

_LAST_UPDATE_DAT_FILENAME = "last_update.dat"


class StockVCPScreener:
    """Stock VCP Screener class."""

    # Set up paths and directories
    toplevel_path = "stock_vcpscreener/"
    csvdatmain_name = toplevel_path + "./db_yfinance/"  # Directory storing stock data csv
    csvdatstooq_name = toplevel_path + "./db_stooq/"  # Directory storing stock data from stooq for crosscheck
    output_path = toplevel_path + "./output/"  # Output directory for storing PDFs
    cdir_path = toplevel_path + "./"  # Current dir with list of tickers
    dsel_info_name = toplevel_path + "./daily_selected_stock_info.csv"  # Output file for the daily stock statistics
    dsel_info_prefix = "selected_stock_"  # Output file for dashboard stock info tables
    source = "yfinance"  # yfinance or stooq

    def __init__(self, in_sel_date: date, in_stock_list: list):
        """Constructor. Required a date and a stock list as input."""
        self._stock_list = in_stock_list
        self._date_to_study = in_sel_date
        print(f"The selected date is {self._date_to_study} ...")

        # Set up a dict used for report
        self._report_dict = self._get_default_report_dict(in_sel_date)

        # A special list of Trust or ETF
        self._special_index_stock_list = {
            "VOO": "brasil",
            "QQQ": "blueskies",
            "DIA": "mike",
            "IWM": "classic",
            "FFTY": "classic",
        }

        self._selected_stock_list = pd.DataFrame(
            columns=[
                "Stock",
                "Index",
                "RS Rating",
                "RS Rating 2",
                "RS Rating 3",
                "50 Day MA",
                "150 Day MA",
                "200 Day MA",
                "52 Week Low",
                "52 week High",
            ]
        )

        self._start_date = self._date_to_study - timedelta(days=365)
        self._end_date = self._date_to_study

        self._selected_stock_rs_rank_list = pd.DataFrame()

    @staticmethod
    def _get_default_report_dict(in_sel_date: date) -> dict:
        """Return a default report dict."""
        return {
            "date": in_sel_date,
            "adv": 0,
            "decl": 0,
            "new_high": 0,
            "new_low": 0,
            "c_20": 0,
            "c_50": 0,
            "s_20_50": 0,
            "s_50_200": 0,
            "s_200_200_20": 0,
            "s_50_150_200": 0,
            "gauge": 0,
            "stocks_fit_condition": 0,
            "index_list": [],
            "stock_ind_list": [],
            "stock_rs_rank_list": [],
            "stock_rs_rating_list": [],
            "breadth_per_list": [],
        }

    def check_directory(self):
        """ Create the directories if not available. Also clean up the output dir by deleting all the pngs and jpgs in
        the directory """
        dir_to_check = [self.csvdatmain_name, self.output_path, self.cdir_path]

        for dir_name in dir_to_check:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        cleanup_dir_jpg_png(self.output_path)

    def check_index_database(self, create=False, update=True):
        """ Check if the index database exist. Create / Update the database """
        if create:
            print("Building Index CSV data")
            create_index_database(self.csvdatmain_name, self.source)

        if update:
            print("Updating Index CSV data")
            update_index_database(self.csvdatmain_name, self.source, self._date_to_study)

    def check_stock_database(self, source, create=False, update=True):
        """ Check if the stock database exist. Create / Update the database """
        if create:
            print("Building Stock CSV data")
            if source == "yfinance":
                create_stock_database(self._stock_list, self.csvdatmain_name, self.source)
            elif source == "stooq":
                create_stock_database(self._stock_list, self.csvdatstooq_name, self.source)

        if update:
            print("Updating Stock CSV data")
            if source == "yfinance":
                update_stock_database(
                    self._stock_list, self.csvdatmain_name, self.source, self._date_to_study
                )  # override=True
            elif source == "stooq":
                update_stock_database(self._stock_list, self.csvdatstooq_name, self.source, self._date_to_study)

    def verify_report_feasibility(self):
        """ Check if we can compile a report of the selected date. Check if the database is updated """
        if not os.path.exists(self.csvdatmain_name + _LAST_UPDATE_DAT_FILENAME):
            print("Cannot find the last update file. Please build the database first")
            return False

        # Read in the database update date
        last_update = pd.read_csv(self.csvdatmain_name + "last_update.dat", header=0)
        last_update["Date"] = pd.to_datetime(last_update["Date"])
        last_update_day = last_update["Date"][0]

        if (last_update_day.date() - self._date_to_study).days < 0:
            print("Please update the stock database.")
            return False

        # Second check - check last day of the GSPC index dataset
        index_last_update_day = get_index_lastday(self.csvdatmain_name)
        if not index_last_update_day:
            print("Cannot find the last update file. Please build the database first")
            return False

        if (index_last_update_day.date() - self._date_to_study).days < 0:
            print("Please wait until yahoo finance update today's data.")
            return False

        return True

    def _check_preconditions(self, overwrite=False):
        if not self.verify_report_feasibility():
            print("Report feasibility check failed. Exiting ...")
            sys.exit(0)

        output_file = os.path.join(self.output_path, f"{str(self._date_to_study)}.pdf")
        if not overwrite and os.path.isfile(output_file):
            print(f"Trade report for {self._date_to_study} already exists. Exiting ...")
            sys.exit(0)

        if self._date_to_study.isoweekday() in {6, 7}:
            print(f"{self._date_to_study} is not a valid trade day. Exiting ...")
            sys.exit(0)

    def select_stock(self, overwrite=False, writecsv=False):
        """ Main method to perform stock selection
        1. verify if it is feasible to generate a report with the selected date
        2. verify if the output pdf already exist, except if overwrite is set to True
        """
        print(f"Working on {self._date_to_study}")

        # Check if it is feasible to work on the selected date
        self._check_preconditions(overwrite=overwrite)

        # Loop through every stock in the provided stock list and compute the stock statistics
        try:
            for ind, stock in enumerate(self._stock_list):
                in_file_name = get_stock_filename(stock)

                if os.path.exists(self.csvdatmain_name + in_file_name):
                    df = pd.read_csv(self.csvdatmain_name + in_file_name, header=0)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    df = df.loc[self._start_date: self._end_date]
                    # df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
                else:
                    # print(f'Cannot read csv. Skipping {stock}')
                    continue

                # Check if the df contains enough number of rows. If yes, proceed, else continue
                if len(df.index) <= 245:
                    # print(f'Not enough data for {stock}. Skipping ...')
                    continue

                current_close = df["Adj Close"][-1]
                ytd_close = df["Adj Close"][-2]
                turnover = df["Volume"][-1] * df["Adj Close"][-1]
                true_range_10d = max(df["Adj Close"][-10:-1]) - min(df["Adj Close"][-10:-1])

                # Compute all the breadth related indices
                if current_close > ytd_close:
                    self._report_dict["adv"] += 1
                    self._report_dict["gauge"] = self._report_dict["gauge"] + turnover
                else:
                    self._report_dict["decl"] += 1
                    self._report_dict["gauge"] = self._report_dict["gauge"] - turnover

                self._report_dict["breadth_per_list"].append((current_close - ytd_close) / ytd_close * 100)

                # Count 52 day new high or low
                if current_close > df["Adj Close"][-250:].max() * 0.97:
                    self._report_dict["new_high"] += 1
                elif current_close < df["Adj Close"][-250:].min() * 1.03:
                    self._report_dict["new_low"] += 1

                # Compute RS ratings of the stock in 3 different ways
                rs_rating, rs_rating2, rs_rating3 = compute_rs_rating(df)

                # Compute SMA and high/low
                df["SMA_20"] = round(df["Adj Close"].rolling(window=20).mean(), 2)
                df["SMA_50"] = round(df["Adj Close"].rolling(window=50).mean(), 2)
                df["SMA_150"] = round(df["Adj Close"].rolling(window=150).mean(), 2)
                df["SMA_200"] = round(df["Adj Close"].rolling(window=200).mean(), 2)
                mov_avg_20 = df["SMA_20"][-1]
                mov_avg_50 = df["SMA_50"][-1]
                mov_avg_150 = df["SMA_150"][-1]
                mov_avg_200 = df["SMA_200"][-1]
                mov_avg_200_20 = df["SMA_200"][-32]  # SMA 200 1 month before (for calculating trending condition)
                low_of_52week = min(df["Adj Close"][-250:])
                high_of_52week = max(df["Adj Close"][-250:])

                # Condition checks
                # Condition 1: Current Price > 150 SMA and > 200 SMA
                condit_1 = current_close > mov_avg_150 > mov_avg_200

                # Condition 2: 50 SMA > 200 SMA
                condit_2 = mov_avg_50 > mov_avg_200
                if condit_2:
                    self._report_dict["s_50_200"] += 1

                # Condition 3: 200 SMA trending up for at least 1 month (ideally 4-5 months)
                condit_3 = mov_avg_200 > mov_avg_200_20
                if condit_3:
                    self._report_dict["s_200_200_20"] += 1

                # Condition 4: 50 SMA > 150 SMA and 150 SMA > 200 SMA
                condit_4 = mov_avg_50 > mov_avg_150 > mov_avg_200
                if condit_4:
                    self._report_dict["s_50_150_200"] += 1

                # Condition 5: Current Price > 50 SMA
                condit_5 = current_close > mov_avg_50
                if condit_5:
                    self._report_dict["c_50"] += 1

                # Condition 6: Current Price is at least 40% above 52 week low (Many of the best are up 100-300% before coming out of consolidation)
                condit_6 = current_close >= (1.40 * low_of_52week)

                # Condition 7: Current Price is within 25% of 52 week high
                condit_7 = current_close >= (0.75 * high_of_52week)

                # Condition 8: Turnover is larger than 2 million
                condit_8 = turnover >= 2000000

                # Condition 9: true range in the last 10 days is less than 8% of current price (consolidation)
                # Should we use the std instead?
                condit_9 = true_range_10d < current_close * 0.08

                # Condition 10: Close above 20 days moving average
                condit_10 = current_close > mov_avg_20
                if condit_10:
                    self._report_dict["c_20"] += 1

                # Condition 11: Current price > $10
                condit_11 = current_close > 10

                # Condition 12: 20 SMA > 50 SMA
                condit_12 = mov_avg_20 > mov_avg_50
                if condit_12:
                    self._report_dict["s_20_50"] += 1

                if (
                        condit_1
                        and condit_2
                        and condit_3
                        and condit_4
                        and condit_5
                        and condit_6
                        and condit_7
                        and condit_8
                        and condit_9
                        and condit_11
                        and condit_12
                ):
                    # condit_6 and condit_7 and condit_8 and condit_9 and condit_10 and \
                    # condit_11 and condit_12:
                    self._report_dict["stocks_fit_condition"] += 1
                    self._selected_stock_list = self._selected_stock_list.append(
                        {
                            "Stock": stock,
                            "Index": ind,
                            "RS Rating": rs_rating,
                            "RS Rating 2": rs_rating2,
                            "RS Rating 3": rs_rating3,
                            "50 Day MA": mov_avg_50,
                            "150 Day MA": mov_avg_150,
                            "200 Day MA": mov_avg_200,
                            "52 Week Low": low_of_52week,
                            "52 week High": high_of_52week,
                        },
                        ignore_index=True,
                    )
                    print(f"/ {stock} matches the requirements! ---")
                else:
                    print(f"{stock}", end=" ")

            # Output stocks that meet the conditions
            print(self._selected_stock_list)
            sel_stock_df = self._selected_stock_list[["Stock", "Index", "RS Rating", "RS Rating 2", "RS Rating 3"]]

            # Compute RS Rank
            self._selected_stock_rs_rank_list = compute_rs_rank(sel_stock_df)

            if writecsv:
                self._selected_stock_list.to_csv(self.cdir_path + "stocks_selected.csv", mode="w")
                self._selected_stock_rs_rank_list.to_csv(self.cdir_path + "stocks_selected_rs_stat.csv", mode="w")

            print(f"Trade day {self._date_to_study} screening completed.")

        except Exception as e:
            print("Error: ")
            print(e)

    def generate_report(self):
        """
        Ultimately select 80.5 percentile of the stocks (selected_stock_list) that matches the conditions
        Generate report PDF and write to the csv file of the daily stock statistics
        """
        _RANK_CRITERIA = 0.805

        # Generate PNGs and JPGs
        stock_namelist = []
        print("Creating PNG plot for:")

        for index, cols in self._selected_stock_rs_rank_list.iterrows():
            try:
                name = cols["Stock"].strip()
                RS_rank = round(cols["RS Rank 3"], 5)
                RS_rating = round(cols["RS Rating 3"], 5)

                if (RS_rank > _RANK_CRITERIA) or (name in self._special_index_stock_list.keys()):
                    shares = yf.Ticker(cols["Stock"])
                    hist = shares.history(start=self._start_date, end=self._end_date, interval="1d")
                    filename = f"{str(RS_rank).ljust(7, '0')}_{name}"
                    titlename = f"{name}   RS Rank: {str(RS_rank).ljust(7, '0')}"
                    outpngfname = self.output_path + "/{}.png".format(filename)
                    outjpgfname = self.output_path + "/{}.jpg".format(filename)
                    kwargs = dict(type="candle", mav=(20, 50, 200), volume=True, figratio=(40, 23), figscale=0.95)

                    stock_namelist.append(name)
                    self._report_dict["stock_ind_list"].append(name)
                    self._report_dict["stock_rs_rank_list"].append(RS_rank)
                    self._report_dict["stock_rs_rating_list"].append(RS_rating)
                    print(f"{name}")

                if name in self._special_index_stock_list.keys():
                    self._report_dict["index_list"].append(name)
                    mpf.plot(
                        hist,
                        **kwargs,
                        style=self._special_index_stock_list[name],
                        title=titlename,
                        savefig=dict(fname=outpngfname, dpi=150, pad_inches=0.1),
                    )
                    convert_png_to_jpg(outpngfname, outjpgfname)
                elif RS_rank > _RANK_CRITERIA:
                    mpf.plot(
                        hist,
                        **kwargs,
                        style="charles",
                        title=titlename,
                        savefig=dict(fname=outpngfname, dpi=150, pad_inches=0.1),
                    )
                    convert_png_to_jpg(outpngfname, outjpgfname)
            except Exception as e:
                print(e)
                print(f"Fail to generate PNG for {name}")

        # Generate the front page and charts, then combine them into a single pdf
        generate_report_output_page(self.output_path, self.cdir_path)
        generate_report_front_page(self._report_dict, stock_namelist, self.cdir_path)
        generate_report_breadth_page(self._report_dict, self._date_to_study, self.cdir_path)
        generate_combined_pdf_report(self.cdir_path, self.output_path, self._date_to_study)

        # Convert report dict to df and update self.dsel_info_name
        df = convert_report_dict_to_df(self._report_dict)
        print(df)

        print(f"Creating dataframe of the trade day {self._date_to_study}")
        if not os.path.exists(self.dsel_info_name):
            df.to_csv(self.dsel_info_name, index=False)
            print(f"Created {self.dsel_info_name}.")
        else:
            org = pd.read_csv(self.dsel_info_name)
            new = org.append(df)
            new["Date"] = pd.to_datetime(new.Date)
            new.set_index("Date", inplace=True)
            new = new[~new.index.duplicated(keep="last")]
            new = new.sort_index()
            new = new.reset_index()
            new.to_csv(self.dsel_info_name, index=False)
            print(f"Updated {self.dsel_info_name}.")

    def generate_dash_csv(self):
        """
        Generate the OHLC for the selected stock. Used for the dashboard
        Select 80.5 percentile of the stocks (selected_stock_list) that matches the conditions
        """
        rank_criteria = 0.805

        out_df = pd.DataFrame(
            columns=[
                "Ticker",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "52 Week Min",
                "52 Week Max",
                "RS Rating",
                "RS Rank",
            ]
        )

        for index, cols in self._selected_stock_rs_rank_list.iterrows():
            try:
                name = cols["Stock"].strip()
                RS_rank = round(cols["RS Rank 3"], 5)
                RS_rating = round(cols["RS Rating 3"], 5)

                if (RS_rank > rank_criteria) or (name in self._special_index_stock_list.keys()):
                    tmp_df = get_stock_data_specific_date(
                        self.csvdatmain_name, name, self._date_to_study, minmax_range=True, percent_change=True
                    )
                    if not isinstance(tmp_df, float):
                        tmp_df["RS Rating"] = RS_rating
                        tmp_df["RS Rank"] = RS_rank
                        out_df = out_df.append(tmp_df)
                    else:
                        print(f"Fail to get stock data for {name}")

            except Exception as e:
                print(f"Fail to generate Dash CSV line for {name} with Error: {e}")

        out_df.to_csv(self.output_path + self.dsel_info_prefix + str(self._date_to_study) + ".csv")
        print(f"Output {self.output_path + self.dsel_info_prefix + str(self._date_to_study) + '.csv'}")


# Start of the program
if __name__ == "__main__":
    # Read in companylist.csv
    data = pd.read_csv("stock_vcpscreener/Tickers.csv", header=0)
    stock_list = list(data.Symbol)

    # Get the last trade day (take yesterday) from current time
    last_weekday = get_last_trade_day().date() - timedelta(days=1)

    # Initiate StockVCPScreener
    svs = StockVCPScreener(last_weekday, stock_list)

    # Checks
    svs.check_directory()
    svs.check_index_database()
    svs.check_stock_database("yfinance", create=False, update=True)  # Use create=True flag to build stock files

    # Select Stock
    # normally with overwrite = False
    svs.select_stock(overwrite=True)

    # Generate report
    svs.generate_report()
    svs.generate_dash_csv()

    # Clean up
    svs.check_directory()
