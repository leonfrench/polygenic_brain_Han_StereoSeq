"""Square bins of cells (scripts/prepare_mouse_bins.py), one more view for the cell app.

Each bin is the pseudobulk of the cells whose centroid lies in it, on one lattice of squares shared by
all slices. It gets the cells' gene-set AUROC from its within-bin ranks, stored per slice in the shard
format of the cell batches (their cell_index holds the bin index).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.rank_io import load_shard
from spatial_app.scoring import auroc_vectorized

MIN_BIN_CELLS = 20  # smaller bins are too shallow to score


def load_bins(path, annotations):
    """The bins' manifest, the bins in drawing order and their annotation levels for the viewer.

    Each of the cell app's levels labels a bin by its cells' most common label (titled as such); the
    added 'bin' level gives the tooltip each bin's cell count and grid position.
    """
    root = Path(path)
    manifest = json.loads((root/'manifest.json').read_text())
    # Earlier builds (square-bins-v1) kept one rank file for all slices; only per-slice ranks can be scored.
    if manifest.get('layout') != 'square-bins-v2' or any('ranks' not in s for s in manifest['slices']):
        raise ValueError('Unsupported bins layout')
    bins = pd.read_parquet(root/'bins.parquet').sort_values('draw_order').reset_index(drop=True)
    if not np.array_equal(np.sort(bins.bin_index.to_numpy()), np.arange(manifest['n_bins'])):
        raise ValueError('Bin indices must be consecutive and unique')
    levels = {k: dict(v, title='Main '+v['title'][0].lower()+v['title'][1:]) for k, v in annotations.items()}
    notes = [f"{n:,} cell{'s' if n != 1 else ''}{', too few to score' if n < MIN_BIN_CELLS else ''}" for n in bins.n_cells]
    levels['bin'] = dict(kind='bin', title='Bin', labels={}, descriptions=dict(zip(bins.bin, notes)))
    return manifest, bins, levels


def bin_scores(path, manifest, bins, query, n_genes):
    """Gene-set AUROC of every bin with at least MIN_BIN_CELLS cells, aligned with bins; NaN for the rest.

    One slice's ranks are read at a time, so they do not stay in server memory.
    """
    values = np.full(manifest['n_bins'], np.nan)
    for entry in manifest['slices']:
        shard = load_shard(Path(path)/entry['ranks'])
        values[shard['cell_index']] = auroc_vectorized(shard, query.target, n_genes, query.background)
    values = values[bins.bin_index.to_numpy()]
    values[bins.n_cells.to_numpy() < MIN_BIN_CELLS] = np.nan
    return values
