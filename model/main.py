import datetime
import sys

import data
import pandas as pd
import training
import utils
import yfinance as yf


def main():
    print(data.snp.snp500_tickers())


if __name__ == "__main__":
    main()
