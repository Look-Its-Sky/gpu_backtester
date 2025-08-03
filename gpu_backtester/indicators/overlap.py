import pandas as pd
import numpy as np
import cudf, cupy 

'''
Bollinger Bands Indicator
'''
def bbands(
    close_prices: cudf.Series | pd.Series, 
    window: int = 20, 
    std: int = 2
) -> cudf.DataFrame:
    if isinstance(close_prices, pd.Series):
        close_prices = cudf.from_pandas(close_prices)

    middle_band = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()

    upper_band = middle_band + (rolling_std * std)
    lower_band = middle_band - (rolling_std * std)

    bband_df = cudf.DataFrame({
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band
    })

    return bband_df 

def sma(
    p: cudf.Series | pd.Series, 
    window: int 
) -> cudf.Series:
    if isinstance(p, pd.Series):
        p = cudf.from_pandas(p)

    sma_series = p.rolling(window=window).mean()

    return sma_series

def ema(
    p: cudf.Series | pd.Series, 
    window: int
) -> cudf.Series:
    if isinstance(p, pd.Series):
        p = cudf.from_pandas(p)

    ema_series = p.ewm(span=window, adjust=False).mean()

    return ema_series

def dema(
    p: cudf.Series | pd.Series, 
    window: int
) -> cudf.Series:
    if isinstance(p, pd.Series):
        p = cudf.from_pandas(p)

    ema1 = p.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()

    dema_series = 2 * ema1 - ema2

    return dema_series 

def tema(
    p: cudf.Series | pd.Series, 
    window: int
) -> cudf.Series:
    if isinstance(p, pd.Series):
        p = cudf.from_pandas(p)

    ema1 = p.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()

    tema_series = 3 * (ema1 - ema2) + ema3

    return tema_series