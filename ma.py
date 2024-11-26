import yfinance as yf
import pandas as pd
import datetime
import sys

# Enums
class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    WHITE = '\033[37m'
    BLACK = '\033[30m'
    YELLOW = '\033[33m'
    CYAN = '\033[36m'
    MAGENTA = '\033[35m'
    RESET = '\033[0m'

# Constatns
MAX_TICKERS = 500
MA_PERIOD = 150
BAR_LENGTH_MAX = 20
COLOR_MA = Colors.CYAN + '|' + Colors.RESET
COLOR_LOW = Colors.RED + '<' + Colors.RESET
COLOR_HIGH = Colors.GREEN + '>' + Colors.RESET

def get_sp500_tickers():
    '''
    Return S&P 500 tickers.
    '''

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return table[0]['Symbol'].tolist()

def high_low_ma_visualization(symb, df, ma):
    high = df['High'].iloc[-1]
    low = df['Low'].iloc[-1]

    dhl = abs(high - low)
    dlma = abs(low - ma)
    dhma = abs(high - ma)

    message = '['
    
    if (ma < low):
        d = dlma + dhl
        sp = (dlma / d) * BAR_LENGTH_MAX
        message += COLOR_MA
        message += ' ' * int(sp)
        message += COLOR_LOW
        message += ' ' * int(BAR_LENGTH_MAX - sp)
        message += COLOR_HIGH
        
    elif (low <= ma <= high):
        d = dhl
        sp = (dlma / d) * BAR_LENGTH_MAX
        message += COLOR_LOW
        message += ' ' * int(sp)
        message += COLOR_MA
        message += ' ' * int(BAR_LENGTH_MAX - sp)
        message += COLOR_HIGH

    elif (ma > high):
        d = dhl + dhma
        sp = (dhl / d) * BAR_LENGTH_MAX
        message += COLOR_LOW
        message += ' ' * int(sp)
        message += COLOR_HIGH
        message += ' ' * int(BAR_LENGTH_MAX - sp)
        message += COLOR_MA

    message += ']'

    return message

def calculate_moving_average(symb, df, period):
    # Check if 'Close' column exists and has df
    n_rows = df['Close'].shape[0]
    if ('Close' not in df or not df['Close'].any() or n_rows < period):
        print(f'Not enought data for {symb}: {n_rows}/{period}')
        return 0.0

    return df['Close'].iloc[-period:].mean()

def print_progress_bar(iteration, total, length=BAR_LENGTH_MAX):
    # Calculate the percentage of completion
    percent = 100 * (iteration / float(total))
    # Calculate the number of filled positions in the bar
    filled_length = int(length * iteration // total)
    # Create the bar string
    bar = '-' * filled_length + ' ' * (length - filled_length)
    # Format the output string
    progress_bar = f"{Colors.MAGENTA}[{bar[:length//2]}{percent:.0f}%{bar[length//2:]}]{Colors.RESET}"
    # Print the bar with carriage return
    sys.stdout.write(f'\r{progress_bar}')
    sys.stdout.flush()

    if percent == 100:
        print()

def check_moving_average(symbols):
    """
    Check if the stock price today hit its MA_PERIOD moving average.

    Args:
        symbols: A list of stock ticker symbols.

    Returns:
        A list of stock symbols that hit their MA_PERIOD moving average today.
    """

    today = datetime.date.today().strftime('%Y-%m-%d')
    results = {}
    for i, s in enumerate(symbols):
        print_progress_bar(i + 1, len(symbols))
        try:
            # Download data (including today)
            df = yf.download(s, period='1y', interval='1d', progress=False)
            df = df.fillna(0) # Replace NaN with 0
            # Calculate the MA_PERIOD moving average
            ma = calculate_moving_average(s, df, MA_PERIOD)
            print_ticker_stats(s, df)
            # Check if low <= ma <= high
            if (df['Low'].iloc[-1] <= ma <= df['High'].iloc[-1]):
                results[s] = df
            
        except Exception as e:
            print(e)
            pass
    return results

def print_ticker_stats(s, df):
    ma = calculate_moving_average(s, df, MA_PERIOD)

    message = f'{Colors.MAGENTA}{s}{Colors.RESET}: '
    message += f'{Colors.RED}Low:{Colors.RESET} {df["Low"].iloc[-1]}, '
    message += f'{Colors.GREEN}High:{Colors.RESET} {df["High"].iloc[-1]}, '
    message += f'{Colors.CYAN}MA-{MA_PERIOD}:{Colors.RESET} {ma}\n'
    message += high_low_ma_visualization(s, df, ma)
    print(message)

    return message

def ma_logic():
    # Get ticker symbols
    # tickers = ['AAPL', 'GOOG', 'MSFT', 'QQQ']
    tickers = get_sp500_tickers()[:MAX_TICKERS]
    print(tickers)
    # text = tickers
    text = ''

    # Find stocks that hit MA_PERIOD-day moving average today
    symbols = check_moving_average(tickers)

    if symbols:
        message = f'The following stocks hit their {MA_PERIOD}-day moving average today:'
        text += message + '\n'
        print(message)
        for s in symbols.keys():
            text += print_ticker_stats(s, symbols[s]) + '\n'
    else:
        message = f'No stocks in the list hit their {MA_PERIOD}-day moving average today.'
        text += message + '\n'
        print(message)

    return text

def main():
    return ma_logic()

if __name__ == '__main__':
    main()
    input("Press Enter to exit...")
