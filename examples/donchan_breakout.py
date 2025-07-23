import pandas as pd
import numpy as np
import cudf, cupy
from pprint import pprint
from backtest.performance import backtest
from indicators.overlap import sma 

def donchan_strategy(
    df: pd.DataFrame | cudf.DataFrame,
    donchian_period: int = 20,
    rr: float = 2.0
):
    rr = float(rr) 

    if isinstance(df, pd.DataFrame):
        df = cudf.from_pandas(df)

    df['upper_donchian'] = df['high'].rolling(window=donchian_period).max().shift(1)
    df['lower_donchian'] = df['low'].rolling(window=donchian_period).min().shift(1)
    df.dropna(inplace=True)
    
    enter_long = (df['close'] > df['upper_donchian']) 
    enter_short = (df['close'] < df['lower_donchian']) 
    df['enter_long'] = enter_long.fillna(False)
    df['enter_short'] = enter_short.fillna(False)

    conflicting_signals_mask = df['enter_long'] & df['enter_short']
    if conflicting_signals_mask.any():
        df.loc[conflicting_signals_mask, ['enter_long', 'enter_short']] = [False, False]

    df['take_profit'] = -1.0
    df['stop_loss'] = -1.0

    long_mask = df['enter_long']
    short_mask = df['enter_short']

    long_risk = df['close'] - df['lower_donchian']
    df['stop_loss'] = df['stop_loss'].mask(long_mask, df['lower_donchian'])
    df['take_profit'] = df['take_profit'].mask(long_mask, df['close'] + long_risk * rr)

    short_risk = df['upper_donchian'] - df['close']
    df['stop_loss'] = df['stop_loss'].mask(short_mask, df['upper_donchian'])
    df['take_profit'] = df['take_profit'].mask(short_mask, df['close'] - short_risk * rr)
   
    return df 

# df = cudf.read_feather('BTC_USDT_USDT-1m-futures.feather')
df = cudf.read_feather('/home/jude/Nextcloud/Fin_data/crypto/BTC_USDT_USDT-1m-futures.feather')
df.set_index('date', inplace=True)

lookback = 60
params = {
    'donchian_period': lookback,
    'rr': 2.0
}

stats, df = backtest(
    strategy_func=donchan_strategy,
    df=df,
    **params
)

pprint(stats)