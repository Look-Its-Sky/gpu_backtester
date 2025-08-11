import numpy as np
import pandas as pd
import cupy 
import cudf
from .performance import *
from typing import Callable

def optimize_2d(
    strategy_func: Callable, 
    df: cudf.DataFrame | pd.DataFrame,
    params: dict,
    target: str,
    commission_pct: float = 0,
    max_bars: int = 999,
    **kwargs
) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        df = cudf.from_pandas(df)

    keys = list(params.keys())
    if len(keys) != 2:
        raise ValueError("This function only supports exactly 2 parameters")
        
    total_runs = len(params[keys[0]]) * len(params[keys[1]]) 
    count = 0
    results = {} 

    for param1 in params[keys[0]]:
        results[float(param1)] = {}

        for param2 in params[keys[1]]:
            print(f'Run {count} of {total_runs}')
            current_params = {keys[0]: param1, keys[1]: param2}
            all_params = {**current_params, **kwargs}
            df_out = strategy_func(df=df, **all_params)
            df_out = add_trade_outcomes(df_out, max_bars=max_bars)
            stats = calculate_performance_stats(df_out, commission_pct=commission_pct)
            
            results[float(param1)][float(param2)] = stats[target]

            count += 1

    return pd.DataFrame(results)
