import pandas as pd
import numpy as np
import cudf, cupy
import gpu_backtester as gbt
import gpu_backtester.indicators as ta
import os
from backtesting import Strategy, Backtest
from pprint import pprint

def atr(high, low, close, period=14):
    tr = pd.DataFrame(np.vstack([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())]).T).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean().values

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
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], period=donchian_period)
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

    df['stop_loss'] = df['stop_loss'].mask(long_mask, df['lower_donchian']) + df['atr']
    df['take_profit'] = df['take_profit'].mask(long_mask, df['close'] * rr) + df['atr']

    short_risk = df['upper_donchian'] - df['close']
    df['stop_loss'] = df['stop_loss'].mask(short_mask, df['upper_donchian'])
    df['take_profit'] = df['take_profit'].mask(short_mask, df['close'] - short_risk * rr)
   
    return df 

class DonchianStrategy(Strategy):
    donchian_period = 20
    rr = 2.0

    def init(self):
        self.upper = self.I(lambda x: pd.Series(x).rolling(self.donchian_period).max().shift(1), self.data.High)
        self.lower = self.I(lambda x: pd.Series(x).rolling(self.donchian_period).min().shift(1), self.data.Low)
        # self.atr = self.I(atr, self.data.df.High, self.data.df.Low, self.data.df.Close, self.donchian_period)
        self.atr = self.I(ta.atr, self.data.df.High, self.data.df.Low, self.data.df.Close)#, self.period)


    def next(self):
        if self.position:
            return

        if self.data.Close[-1] > self.upper[-1]:
            self.buy(
                sl=self.lower[-1] - self.atr[-1],
                tp=self.data.Close[-1] + (self.data.Close[-1] - self.lower[-1]) * self.rr
            )
        elif self.data.Close[-1] < self.lower[-1]:
            self.sell(
                sl=self.upper[-1] + self.atr[-1],
                tp=self.data.Close[-1] - (self.upper[-1] - self.data.Close[-1]) * self.rr
            )

if __name__ == "__main__":
    df = pd.read_feather('BTC_USDT_USDT-1m-futures.feather')
    df.columns = [col.capitalize() for col in df.columns]
    df.set_index('Date', inplace=True)

    bt = Backtest(df, DonchianStrategy, cash=100_000, commission=.002)
    stats = bt.run()
    pprint(stats)
