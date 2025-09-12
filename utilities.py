import yfinance as yf
import pandas as pd

def use_sm(s, m):
    pass


def test_yf():
    dat = yf.Ticker("MSFT")
    dat.info
    dat.calendar
    dat.analyst_price_targets
    dat.quarterly_income_stmt
    dat.history(period='1mo')
    dat.option_chain(dat.options[0]).calls
    print(dat.history.head())

def main():
    test_yf()


if __name__ == "__main__":
    main()




