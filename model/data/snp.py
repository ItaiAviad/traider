import pandas as pd


def snp500_tickers():
    """
    Return S&P 500 tickers
    """

    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return table[0]["Symbol"].tolist()
