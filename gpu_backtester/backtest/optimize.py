import numpy as np
import pandas as pd
import cupy 
import cudf
import itertools

from .performance import *

from typing import Callable

'''
Takes in a dict of parameters 
And runs a grid opt

Example
params = {
    'lookback': np.arange(10, 100, 10),
    'rr': np.arange(1.0, 3.0, 0.5)
}

lookback is param1 
rr is param2

For now this will only be 2d optimization
'''

def optimize(
    strategy_func: Callable, 
    df: cudf.DataFrame | pd.DataFrame,
    params: dict,
    target: str,
    commission_pct: float = 0,
    max_bars: int = 999,
    **kwargs
) -> dict:
    if isinstance(df, pd.DataFrame):
        df = cudf.from_pandas(df)

    def bt(param1, param2, df):
        df = strategy_func(df=df, **kwargs)
        df = add_trade_outcomes(df, max_bars=max_bars)

        return calculate_performance_stats(df, commission_pct=commission_pct)

    keys = list(params.keys())
    results = {} 

    for param1 in params[keys[0]]:
        results[float(param1)] = {}
        for param2 in params[keys[1]]:
            stats = bt(param1, param2, df)[target]

            results[float(param1)][float(param2)] = stats

    return pd.DataFrame(results) 
