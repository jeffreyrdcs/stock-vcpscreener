# Stock-vcpscreener

Stock-vcpscreener is a simple stock screener that calculates various market breadth indicators and selects US stocks on a daily basis based on multiple criteria. The criteria are applied to the simple moving averages and price performance of the stocks over the last year. The goal is to look for a volatility contraction pattern in the market, a specific chart pattern that suggests the stock may move into an uptrend in the near future.

This project started off as a code refactoring practice based on a command-line stock screener I saw on an online forum [(ref: carlam.net)](https://carlam.net/), which in turn seems to be inspired from this [medium article](https://towardsdatascience.com/making-a-stock-screener-with-python-4f591b198261) after a bit of googling. This project now forms the backbone of a Dash app, which visualizes the selected stocks and compiles a daily US stock market analysis report.


## Usage

Everything is done through the StockVCPScreener class. To initate:
```
svs = StockVCPScreener(selected_date[datetime.date], stock_list[list])
```

To download the stock data to the directory (currently supports yahoo finance or stooq):
```
svs.check_stock_database('yfinance', create=True)
```

To check the output directory and update the directories containing the stock data
```
svs.check_directory()
svs.check_index_database()
svs.check_stock_database('yfinance')
```

To run the stock selection:
```
svs.select_stock()
```

To generate a report and the csv files necessary for the Dash app:
```
svs.generate_report()
svs.generate_dash_csv()
```

## Visualization

The Dash app can be accessed at the link below:

https://jeffreyrdcs.pythonanywhere.com


## To-do

* Complete the test cases
* Add a method to validate and fix the database directories



