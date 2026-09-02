"""Data loading utilities for all three datasets."""
import os
import pandas as pd
import numpy as np
from config import PIMA_PATH, NHANES_FILES, BRFSS_PATH, BRFSS_ASC


def load_pima(path=None):
    """Load raw PIMA Indians Diabetes Dataset."""
    path = path or PIMA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"PIMA file not found: {path}")
    df = pd.read_csv(path)
    return df


def load_nhanes_xpt(files_dict=None):
    """Load NHANES XPT files and return dictionary of DataFrames."""
    files = files_dict or NHANES_FILES
    data = {}
    for key, path in files.items():
        if os.path.exists(path):
            data[key] = pd.read_sas(path, format='xport', encoding='utf-8')
        else:
            print(f"Warning: NHANES {key} not found at {path}")
            data[key] = None
    return data


def load_brfss(path=None, chunk_size=50000, return_audit=False):
    """
    Load the official CDC BRFSS 2015 XPT file in chunks.

    No CSV/Kaggle fallback and no row sampling.
    All records are read and concatenated.

    Returns
    -------
    df : pandas.DataFrame
        Full BRFSS dataset.
    audit : list, optional
        Audit records when return_audit=True.
    """
    path = path or BRFSS_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Official BRFSS XPT file not found: {path}"
        )

    print(f"Loading official CDC BRFSS XPT: {path}")
    print(f"Chunk size: {chunk_size:,}")
    print("Full-file processing enabled; no rows will be sampled.")

    chunks = []
    audit = []
    total_rows = 0
    chunk_number = 0

    try:
        with pd.read_sas(
            path,
            format="xport",
            encoding="utf-8",
            iterator=True
        ) as reader:

            while True:
                try:
                    chunk = reader.read(chunk_size)
                except StopIteration:
                    # Normal end-of-file condition for some XPT readers.
                    break

                if chunk is None or chunk.empty:
                    break

                chunk_number += 1
                rows = len(chunk)
                total_rows += rows

                chunks.append(chunk)

                print(
                    f"BRFSS chunk {chunk_number}: "
                    f"{rows:,} records x {chunk.shape[1]} variables "
                    f"(cumulative: {total_rows:,})"
                )

                audit.append({
                    "Dataset": "BRFSS",
                    "Stage": f"Chunk {chunk_number}",
                    "Operation": "Full-file XPT ingestion",
                    "Records_Before": rows,
                    "Records_After": rows,
                    "Removed": 0,
                    "Variables_Before": chunk.shape[1],
                    "Variables_After": chunk.shape[1],
                })

    except Exception as e:
        import traceback
        print("\nBRFSS XPT READ ERROR:")
        traceback.print_exc()
        raise RuntimeError(
            f"Failed while reading official BRFSS XPT file: {e}"
        ) from e

    if not chunks:
        raise RuntimeError(
            "BRFSS XPT reader returned no records."
        )

    df = pd.concat(chunks, ignore_index=True)

    # Hard integrity check
    expected_minimum = 441456

    if len(df) < expected_minimum:
        raise RuntimeError(
            f"BRFSS integrity check failed: "
            f"expected at least {expected_minimum:,} raw records, "
            f"but loaded {len(df):,}."
        )

    print(
        f"BRFSS FULL FILE loaded: "
        f"{len(df):,} records x {df.shape[1]} variables"
    )

    audit.insert(0, {
        "Dataset": "BRFSS",
        "Stage": "Raw ingestion",
        "Operation": "Official CDC LLCP2015.XPT full-file ingestion",
        "Records_Before": len(df),
        "Records_After": len(df),
        "Removed": 0,
        "Variables_Before": df.shape[1],
        "Variables_After": df.shape[1],
    })

    if return_audit:
        return df, audit

    return df