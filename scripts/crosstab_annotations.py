"""Cross-tabulate transferred cell types against CCF regions for a prepared dataset.

  .conda-env/bin/python scripts/crosstab_annotations.py [--data data/mouse1] [--out analysis]

Writes:
  celltype_region_counts.csv  long counts: slice x hemisphere x region levels x cell type levels
  subclass_by_division.csv    cell subclass x (slice, brain division) cell counts
  subclass_by_area.csv        cell subclass x (slice, brain area) cell counts
  marker_detection.csv        % of cells detecting OB / cortical markers, by subclass, slice and OLF vs elsewhere
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ['Tbx21', 'Cdhr1', 'Slc17a7', 'Satb2', 'Neurod6', 'Gad1']
CELL = ['cell_class', 'cell_subclass', 'cell_cluster']
REGION = ['hemisphere', 'region_division', 'region_area', 'region']


def wide(cells, region_column):
    table = pd.crosstab(cells.cell_subclass, [cells.slice, cells[region_column]])
    table[('All', 'All')] = table.sum(axis=1)
    return table


def marker_detection(root, manifest, cells, genes, markers):
    """Fraction of cells with at least one UMI of each marker, from the stored sparse ranks."""
    index = {s: int(genes.index[genes.symbol.eq(s)][0]) for s in markers if genes.symbol.eq(s).any()}
    detected = {s: np.zeros(len(cells), bool) for s in index}
    for path in [b for s in manifest['slices'] for b in s['batches']]:
        with np.load(root/path, allow_pickle=False) as z:
            rows = np.repeat(z['cell_index'], np.diff(z['indptr'].astype(np.int64)))
            for symbol, gene in index.items():
                detected[symbol][rows[z['gene_index'] == gene]] = True
    location = np.where(cells.region_division.eq('OLF'), 'OLF', 'outside OLF')
    keys = [cells.cell_subclass, cells.slice, pd.Series(location, name='location')]
    table = pd.DataFrame({f'{s}_pct_detected': v for s, v in detected.items()}).groupby(keys).mean().mul(100).round(1)
    table.insert(0, 'cells', pd.Series(1, index=cells.index).groupby(keys).sum())
    table['median_umi'] = cells.n_umi.groupby(keys).median()
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT/'data/mouse1')
    parser.add_argument('--out', type=Path, default=ROOT/'analysis')
    parser.add_argument('--markers', nargs='*', default=MARKERS)
    args = parser.parse_args()
    manifest = json.loads((args.data/'manifest.json').read_text())
    cells = pd.read_parquet(args.data/'cells.parquet').sort_values('cell_index').reset_index(drop=True)
    genes = pd.read_parquet(args.data/'genes.parquet').sort_values('gene_index').reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    long = cells.groupby(['slice', *REGION, *CELL], observed=True).size().rename('n_cells').reset_index()
    long.to_csv(args.out/'celltype_region_counts.csv', index=False)
    wide(cells, 'region_division').to_csv(args.out/'subclass_by_division.csv')
    wide(cells, 'region_area').to_csv(args.out/'subclass_by_area.csv')
    marker_detection(args.data, manifest, cells, genes, args.markers).to_csv(args.out/'marker_detection.csv')
    print(f'Wrote 4 tables for {len(cells):,} cells in {manifest["n_slices"]} slices to {args.out}')


if __name__ == '__main__':
    main()
