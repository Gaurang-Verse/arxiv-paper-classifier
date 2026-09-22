"""Draw a reproducible random subsample from the raw arXiv metadata file.

Sampling is done in a single streaming pass: we pre-select which global row
indices to keep (using a seeded RNG), then walk the file in chunks and pull
out only those rows. This keeps memory flat regardless of file size, and is
exactly reproducible given the same seed and the same input file.
"""

import numpy as np
import pandas as pd


def sample_rows(
    path: str,
    expected_total_rows: int,
    sample_size: int,
    seed: int,
    chunksize: int,
    keep_columns: list[str],
) -> pd.DataFrame:
    """Return a random subsample of `sample_size` rows from the JSON-lines file.

    Raises ValueError if the file's actual row count differs from
    `expected_total_rows`, since that would mean the dataset changed and the
    sample is no longer the one our documented results refer to.
    """
    if sample_size > expected_total_rows:
        raise ValueError(
            f"sample_size ({sample_size}) exceeds expected_total_rows "
            f"({expected_total_rows})"
        )

    rng = np.random.default_rng(seed)
    keep_indices = np.sort(
        rng.choice(expected_total_rows, size=sample_size, replace=False)
    )

    selected_chunks = []
    offset = 0

    reader = pd.read_json(path, lines=True, chunksize=chunksize, dtype={"id": str})
    for chunk in reader:
        n_rows = len(chunk)

        # Which of our pre-selected global indices fall inside this chunk?
        start = np.searchsorted(keep_indices, offset, side="left")
        end = np.searchsorted(keep_indices, offset + n_rows, side="left")

        if end > start:
            local_positions = keep_indices[start:end] - offset
            selected_chunks.append(chunk.iloc[local_positions][keep_columns])

        offset += n_rows

    if offset != expected_total_rows:
        raise ValueError(
            f"Row count mismatch: file has {offset} rows but config expects "
            f"{expected_total_rows}. The dataset appears to have changed — "
            f"re-run validation and update configs/data.yaml before sampling."
        )

    return pd.concat(selected_chunks, ignore_index=True)