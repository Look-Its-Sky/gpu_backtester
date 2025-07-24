import pandas as pd 
import numpy as np
import cudf, cupy

''' Relative Strength Index (RSI) '''
def rsi(
    close: cudf.Series | pd.Series,
    period: int = 14
) -> cudf.Series | pd.Series:
    # Calculate price changes
    delta = close.diff(1)
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    # Calculate the exponential moving average (Wilder's smoothing)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate the RSI
    # The formula naturally handles the case where avg_loss is 0 (RSI becomes 100)
    rsi_series = 100 - (100 / (1 + rs))
    
    return rsi_series.fillna(0)