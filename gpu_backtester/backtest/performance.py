from numba import njit, cuda
from typing import Callable
import pandas as pd
import numpy as np
import cudf
import cupy
import gc
import pynvml

def _get_vram_info():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.free, info.total

def _df_fits_in_vram(df, safety_margin=0.8):
    free_vram, _ = _get_vram_info()
    df_size_bytes = df.memory_usage(deep=True).sum()
    return df_size_bytes < free_vram * safety_margin

''' Just a wrapper around the provided strategy '''
def backtest(
    strategy_func: Callable, 
    df: cudf.DataFrame | pd.DataFrame, 
    commission_pct: float = 0,
    max_bars: int = 999,
    **kwargs
) -> dict:

    # Chunk the df if it doesn't fit in VRAM
    if not _df_fits_in_vram(df):
        processed_chunks = []
        chunksize = 1_000_000  # 1 million rows per chunk
        
        # Ensure df is a pandas DataFrame for chunking
        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()

        n_chunks = (len(df) - 1) // chunksize + 1

        for i in range(n_chunks):
            start = i * chunksize
            end = start + chunksize
            chunk_pd = df.iloc[start:end]
            
            # Process chunk on GPU
            chunk_cudf = cudf.from_pandas(chunk_pd)
            chunk_cudf = strategy_func(df=chunk_cudf, **kwargs)
            chunk_cudf = add_trade_outcomes(chunk_cudf, max_bars=max_bars)
            
            # Keep only columns needed for stats, and store as pandas df
            cols_to_keep = ['close', 'outcome', 'exit_price', 'enter_long']
            processed_chunks.append(chunk_cudf[cols_to_keep].to_pandas())
            
            # Clean up GPU memory
            del chunk_cudf
            gc.collect()
            cupy.get_default_memory_pool().free_all_blocks()

        # Combine chunks and calculate stats on the full dataset
        full_processed_df = pd.concat(processed_chunks)
        stats = calculate_performance_stats(full_processed_df, commission_pct=commission_pct)
        
        return stats, None

    df = strategy_func(df=df, **kwargs)
    df = add_trade_outcomes(df, max_bars=max_bars)
    stats = calculate_performance_stats(df, commission_pct=commission_pct)
    gc.collect()  

    return stats, df

''' CUDA Kernel to label the outcome and exit price of trades ''' 
@cuda.jit
def label_outcomes_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    enter_long: np.ndarray,
    enter_short: np.ndarray,
    take_profit: np.ndarray,
    stop_loss: np.ndarray,
    outcomes: np.ndarray,
    exit_prices: np.ndarray, 
    trade_duration: int
):
    i = cuda.grid(1)
    if i >= len(close) - trade_duration:
        return

    # --- Long Trade Outcome ---
    if enter_long[i]:
        entry_price = close[i]
        for j in range(1, trade_duration + 1):
            current_bar_idx = i + j

            if high[current_bar_idx] >= take_profit[i]:
                outcomes[i] = 1
                exit_prices[i] = take_profit[i]
                return

            if low[current_bar_idx] <= stop_loss[i]:
                outcomes[i] = -1
                exit_prices[i] = stop_loss[i] 
                return

        # Timeout exit
        exit_prices[i] = close[i + trade_duration] 
        outcomes[i] = 1 if exit_prices[i] > entry_price else -1
        
    # --- Short Trade Outcome ---
    elif enter_short[i]:
        entry_price = close[i]
        for j in range(1, trade_duration + 1):
            current_bar_idx = i + j
            
            if low[current_bar_idx] <= take_profit[i]:
                outcomes[i] = 1
                exit_prices[i] = take_profit[i] 
                return
            
            if high[current_bar_idx] >= stop_loss[i]:
                outcomes[i] = -1
                exit_prices[i] = stop_loss[i] 
                return

        # Timeout exit
        exit_prices[i] = close[i + trade_duration]
        outcomes[i] = 1 if exit_prices[i] < entry_price else -1

'''
Wrapper function to apply the GPU-based outcome labeling.
This version now also returns the exit price for each trade.
'''

def add_trade_outcomes(
    df: pd.DataFrame, 
    max_bars: int = 999
) -> pd.DataFrame:
    high_np = df['high'].values
    low_np = df['low'].values
    close_np = df['close'].values
    enter_long_np = df['enter_long'].values
    enter_short_np = df['enter_short'].values
    take_profit_np = df['take_profit'].values
    stop_loss_np = df['stop_loss'].values

    outcomes_np = np.zeros(len(df), dtype=np.int8)
    exit_prices_np = np.zeros(len(df), dtype=np.float64) 

    high_gpu = cuda.to_device(high_np)
    low_gpu = cuda.to_device(low_np)
    close_gpu = cuda.to_device(close_np)
    enter_long_gpu = cuda.to_device(enter_long_np)
    enter_short_gpu = cuda.to_device(enter_short_np)
    take_profit_gpu = cuda.to_device(take_profit_np)
    stop_loss_gpu = cuda.to_device(stop_loss_np)
    outcomes_gpu = cuda.to_device(outcomes_np)
    exit_prices_gpu = cuda.to_device(exit_prices_np)

    threads_per_block = 128
    blocks_per_grid = (len(df) + (threads_per_block - 1)) // threads_per_block

    label_outcomes_kernel[blocks_per_grid, threads_per_block](
        high_gpu, low_gpu, close_gpu,
        enter_long_gpu, enter_short_gpu,
        take_profit_gpu, stop_loss_gpu,
        outcomes_gpu,
        exit_prices_gpu,
        max_bars
    )

    outcomes_gpu.copy_to_host(outcomes_np)
    exit_prices_gpu.copy_to_host(exit_prices_np) 

    df['outcome'] = outcomes_np
    df['exit_price'] = exit_prices_np 

    return df

@njit
def _calculate_annualization_periods(index_values_as_int: np.ndarray) -> float:
    """Calculates the number of trading periods in a year."""
    timedelta_ns = index_values_as_int[-1] - index_values_as_int[0]
    
    # Convert nanoseconds to seconds
    seconds = timedelta_ns / 1_000_000_000
    
    if seconds > 0:
        years = seconds / (365.25 * 24 * 60 * 60)
        if years > 0:
            return len(index_values_as_int) / years
    
    return (252 * 6.5 * 60)

def calculate_performance_stats(
    df: cudf.DataFrame,
    commission_pct: float = 0.0,
    risk_free_rate: float = 0.0
) -> dict:
    if isinstance(df, pd.DataFrame):
        df = cudf.from_pandas(df)

    trades = df[df['outcome'] != 0].copy()
    if trades.empty:
        return {
            "Total Trades": 0,
            "Win Rate (%)": 0,
            "Total Return (%)": 0,
            "Sharpe Ratio": 0, 
            "Sortino Ratio": 0,
            "Max Drawdown (%)": 0,
            "Profit Factor": 0,
            "Annualization Periods": 0
        }

    entry_price = trades['close']
    long_returns = (trades['exit_price'] - entry_price) / entry_price
    short_returns = (entry_price - trades['exit_price']) / entry_price
    gross_returns = cupy.where(trades['enter_long'], long_returns, short_returns)
    trade_returns = gross_returns - commission_pct

    returns_df = cudf.DataFrame({'trade_return': trade_returns}, index=trades.index)
    merged_df = df.merge(returns_df, how='left', left_index=True, right_index=True)
    strategy_returns = merged_df['trade_return'].fillna(0.0)

    # --- Annualization ---
    periods_per_year = _calculate_annualization_periods(df.index.to_pandas().values.astype(np.int64))

    # --- Aggregate Stats ---
    total_trades = len(trades)
    wins = (trade_returns > 0).sum().item()
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    gross_profit = strategy_returns[strategy_returns > 0].sum()
    gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else cupy.inf

    # --- Portfolio-level stats ---
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # --- Risk-adjusted returns ---
    per_period_risk_free_rate = (1 + risk_free_rate)**(1/periods_per_year) - 1
    excess_returns = strategy_returns - per_period_risk_free_rate
    mean_excess_return = excess_returns.mean()
 
    std_dev = excess_returns.std()
    sharpe_ratio = (mean_excess_return / std_dev) * cupy.sqrt(periods_per_year) if std_dev > 0 else 0.0

    downside_dev_returns = excess_returns.copy()
    downside_dev_returns[downside_dev_returns > 0] = 0.0
    downside_deviation = cupy.sqrt((downside_dev_returns**2).mean())

    sortino_ratio = (mean_excess_return / downside_deviation) * cupy.sqrt(periods_per_year) if downside_deviation > 0 else 0.0
        
    return {
        "Total Trades": total_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Total Return (%)": round(total_return.item(), 2),
        "Profit Factor": round(float(profit_factor), 2) if profit_factor != cupy.inf else 'inf',
        "Max Drawdown (%)": round(max_drawdown.item(), 2),
        "Sharpe Ratio": round(sharpe_ratio.item(), 2),
        "Sortino Ratio": round(sortino_ratio.item(), 2),
        "Annualization Periods": round(periods_per_year)
    }
