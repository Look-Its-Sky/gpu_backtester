import pandas as pd
import numpy as np
from pynvml import *

"""
Gets the available GPU VRAM in MiB using pynvml.
Assumes you are using the GPU at index 0.
"""
def get_available_vram_mb():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()

        # Convert to MiB
        return info.free / (1024**2)

    except NVMLError as error:
        print(f"Warning: Could not query GPU VRAM. Error: {error}.")
        print("Falling back to a default of 2048 MiB.")

        return 2048

def get_dataframe_chunks(df, vram_safety_margin=0.8):
    """
    Generator function to yield DataFrame chunks that fit in available VRAM.
    """
    available_vram_mb = get_available_vram_mb()
    safe_vram_allocation_mb = available_vram_mb * vram_safety_margin

    total_bytes = df.memory_usage(deep=True).sum()
    bytes_per_row = total_bytes / len(df)

    if bytes_per_row == 0:
        rows_per_chunk = len(df)
    else:
        rows_per_chunk = int((safe_vram_allocation_mb * 1024**2) / bytes_per_row)

    if rows_per_chunk == 0:
        raise ValueError("Not enough VRAM to process even a single row of the DataFrame.")

    n_chunks = int(np.ceil(len(df) / rows_per_chunk))

    print(f"Available VRAM: {available_vram_mb:.2f} MiB")
    print(f"Safe VRAM Allocation ({vram_safety_margin*100}%): {safe_vram_allocation_mb:.2f} MiB")
    print(f"DataFrame total size: {total_bytes / (1024**2):.2f} MiB")
    print(f"Bytes per row: {bytes_per_row:.2f}")
    print(f"Calculated rows per chunk: {rows_per_chunk}")
    print(f"Number of chunks to process: {n_chunks}")

    for i in range(n_chunks):
        start_index = i * rows_per_chunk
        end_index = min((i + 1) * rows_per_chunk, len(df))
        yield df.iloc[start_index:end_index]


if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. Create a large sample DataFrame
    print("Creating a sample DataFrame...")
    num_rows = 20_000_000
    data = {
        'price': np.float32(np.random.rand(num_rows)),
        'volume': np.int32(np.random.randint(100, 1000, num_rows)),
        'symbol': ['SYM' + str(i % 500) for i in range(num_rows)]
    }
    my_dataframe = pd.DataFrame(data)
    print("Sample DataFrame created.")

    # 2. Process the DataFrame in VRAM-aware chunks
    try:
        chunk_generator = get_dataframe_chunks(my_dataframe, vram_safety_margin=0.8)
        
        for i, df_chunk in enumerate(chunk_generator):
            print(f"\nProcessing chunk {i+1} (rows {df_chunk.index[0]} to {df_chunk.index[-1]})")
            print(f"Chunk size: {len(df_chunk)} rows")
            
            # --- YOUR GPU PROCESSING LOGIC GOES HERE ---
            # e.g., import cudf
            # gdf_chunk = cudf.from_pandas(df_chunk)
            # print(f"cuDF chunk VRAM usage: {gdf_chunk.memory_usage().sum() / (1024**2):.2f} MiB")
            # ... run your backtest on gdf_chunk ...
            # -----------------------------------------

    except (ValueError, NVMLError) as e:
        print(f"An error occurred: {e}")
