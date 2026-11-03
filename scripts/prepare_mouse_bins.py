"""Stereo-seq slices as square bins of cells: each bin is the pseudobulk of the cells inside it (experimental).

The prepared cell dataset (--cells, from prepare_mouse_cells.py) gives the cells, their registered centroids,
labels and the gene universe. It stores ranks, not counts, so each slice's raw spot table in --raw is read
again, streamed from the gzip file. Cells are assigned by centroid to the squares of one lattice shared by all
slices (they share the registered frame), so every bin has the same size, and a row and column mean the same
position in every slice; smaller slices simply have fewer bins. --bin-um sets the side in µm; --bins sets it so
that the widest slice spans that many squares. A bin's counts are the summed retained-gene UMIs of its cells,
ranked within the bin as the cells are.

Checks: every cell's retained UMIs equal n_umi_retained in the cell dataset, each slice's bin gene totals equal
its cell class pseudobulk, and --check-bins bins per slice match scipy.stats.rankdata and a pairwise AUROC.

  .conda-env/bin/python scripts/prepare_mouse_bins.py --bin-um 178
  .conda-env/bin/python scripts/prepare_mouse_bins.py --slices T319 --bins 25
"""
from pathlib import Path
import argparse, datetime, json, math, os, platform, shutil, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import scipy

from prepare_mouse_cells import LEVELS, sha256, rank_rows, validate_rows

ROOT = Path(__file__).resolve().parents[1]
UM_PER_UNIT = 0.5  # registered Stereo-seq spot units
PARENT = {'cell_cluster': 'cell_subclass', 'cell_subclass': 'cell_class'}


def bin_counts(path, cell_ids, cell_bin, n_bins, lookup, n_genes, block_size=1 << 28):
    """Summed UMIs per (bin, gene) and per cell for the listed cells and mapped genes of one spot table.

    cell_ids are the source cell labels and cell_bin their bins; lookup maps source symbols to gene_index.
    """
    position = np.full(int(cell_ids.max())+1, -1, np.int64)
    position[cell_ids] = np.arange(len(cell_ids))
    names, gene_of = pa.array(list(lookup)), np.fromiter(lookup.values(), np.int64, len(lookup))
    sums, cell_umi, rows, umi_total = np.zeros(n_bins*n_genes), np.zeros(len(cell_ids)), 0, 0
    convert = pv.ConvertOptions(include_columns=['gene', 'umi_count', 'cell_label'],
                                column_types={'gene': pa.string(), 'umi_count': pa.int64(), 'cell_label': pa.int64()})
    with pa.input_stream(str(path), compression='gzip') as stream:
        reader = pv.open_csv(stream, read_options=pv.ReadOptions(block_size=block_size),
                             parse_options=pv.ParseOptions(delimiter='\t'), convert_options=convert)
        for batch in reader:
            label, umi = batch.column('cell_label').to_numpy(), batch.column('umi_count').to_numpy()
            rows += len(label); umi_total += int(umi.sum())
            cell = np.full(len(label), -1)
            inside = (label > 0) & (label < len(position))  # cell_label 0 is a spot outside any cell
            cell[inside] = position[label[inside]]
            gene = pc.fill_null(pc.index_in(batch.column('gene'), value_set=names), -1).to_numpy()
            keep = (cell >= 0) & (gene >= 0)
            cell, gene, umi = cell[keep], gene_of[gene[keep]], umi[keep]
            cell_umi += np.bincount(cell, weights=umi, minlength=len(cell_ids))
            sums += np.bincount(cell_bin[cell]*n_genes + gene, weights=umi, minlength=n_bins*n_genes)
    return sums.reshape(n_bins, n_genes), cell_umi, rows, umi_total


def main_labels(codes, labels, local, n_bins):
    """Each bin's most common label (ties: first in sorted order) and that label's fraction of the bin's cells."""
    tally = np.bincount(local*len(labels) + codes, minlength=n_bins*len(labels)).reshape(n_bins, len(labels))
    top = tally.argmax(axis=1)
    return labels[top], tally[np.arange(n_bins), top]/tally.sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/Han et al.')
    parser.add_argument('--cells', type=Path, default=ROOT/'data/mouse1', help='Prepared cell dataset (prepare_mouse_cells.py).')
    parser.add_argument('--slices', nargs='*', help='Slice IDs of the cell dataset, e.g. T319 T294. Default: all of them.')
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument('--bin-um', type=float, help='Side of the squares in µm, rounded to whole 0.5 µm units.')
    size.add_argument('--bins', type=int, help='Squares across the longer side of the widest slice; sets the side for all slices.')
    parser.add_argument('--check-bins', type=int, default=64, help='Bins per slice checked against scipy.stats.rankdata.')
    parser.add_argument('--out', type=Path, help='Default: the cell dataset folder name with -bins<µm>um.')
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--force', action='store_true', help='Replace an existing output directory.')
    args = parser.parse_args()
    started = time.perf_counter()
    source = json.loads((args.cells/'manifest.json').read_text())
    entries = {s['id']: s for s in source['slices']}
    requested = args.slices or list(entries)
    if any(s not in entries for s in requested):
        raise SystemExit(f'Slices not in {args.cells}: {[s for s in requested if s not in entries]}')
    slices = [s for s in entries if s in requested]  # the cell dataset's anterior-to-posterior order

    genes = pd.read_parquet(args.cells/'genes.parquet').sort_values('gene_index').reset_index(drop=True)
    g = len(genes)
    lookup = {name: i for i, names in enumerate(genes.source_genes) for name in names.split('|')}
    cells = pd.read_parquet(args.cells/'cells.parquet', columns=['cell_index', 'slice', 'cell_id', 'rx', 'ry', 'n_umi_retained', *LEVELS])
    cells = cells[cells.slice.isin(slices)].sort_values('cell_index').reset_index(drop=True)
    if args.bin_um:
        side = round(args.bin_um/UM_PER_UNIT)
    else:
        extent = max(max(p.rx.max()-p.rx.min(), p.ry.max()-p.ry.min()) for _, p in cells.groupby('slice'))
        side = math.ceil(extent/args.bins)
    if side < 1:
        raise SystemExit('Bins must be at least 0.5 µm')
    bin_um = side*UM_PER_UNIT
    # One lattice for all slices, from the top left of their shared registered frame. Its side is a whole number
    # of units, so bin centres are exactly one side apart and neighbouring squares meet without gaps.
    origin = (math.floor(cells.rx.min()), math.floor(cells.ry.min()))
    n_rows, n_cols = int((cells.ry.max()-origin[1])//side) + 1, int((cells.rx.max()-origin[0])//side) + 1

    out = args.out or args.cells.with_name(f'{args.cells.name}-bins{bin_um:g}um')
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f'{out} is not empty; pass --force to replace it')
    staging = out.with_name(out.name + '.staging')
    if staging.exists():
        shutil.rmtree(staging)
    for folder in ('counts', 'ranks', 'pseudobulk'):
        (staging/folder).mkdir(parents=True)
    manifest_files = []
    def record(path):
        manifest_files.append(dict(path=str(path.relative_to(staging)), bytes=path.stat().st_size, sha256=sha256(path)))

    rng = np.random.default_rng(args.seed)
    tables, cell_bins, slice_entries, validated, offset = [], [], [], 0, 0
    for slice_id in slices:
        t = time.perf_counter()
        entry = entries[slice_id]
        path = args.raw/entry['source']
        if path.stat().st_size != entry['source_bytes']:
            raise ValueError(f'{path} differs from the file the cell dataset was built from')
        part = cells[cells.slice.eq(slice_id)]
        row = ((part.ry.to_numpy()-origin[1])//side).astype(np.int64)
        col = ((part.rx.to_numpy()-origin[0])//side).astype(np.int64)
        occupied, local = np.unique(row*n_cols + col, return_inverse=True)
        n = len(occupied)
        sums, cell_umi, rows, umi = bin_counts(path, part.cell_id.to_numpy(), local, n, lookup, g)
        # The same cells and genes as the cell dataset: per-cell totals, then per-gene totals of the slice.
        if (rows, umi) != (entry['rows'], entry['umi']):
            raise ValueError(f'{slice_id}: read {rows:,} rows and {umi:,} UMIs, the cell dataset {entry["rows"]:,} and {entry["umi"]:,}')
        np.testing.assert_array_equal(cell_umi, part.n_umi_retained.to_numpy())
        with np.load(args.cells/entry['pseudobulk'], allow_pickle=False) as z:
            np.testing.assert_array_equal(sums.sum(axis=0), z['cell_class_counts'].sum(axis=0, dtype=np.float64))
        if sums.max(initial=0) >= 2**32:
            raise ValueError('Bin counts exceed uint32')
        counts = sums.astype(np.uint32)
        del sums
        r, c = occupied // n_cols, occupied % n_cols
        table = pd.DataFrame(dict(slice=slice_id, bin=[f'{slice_id} r{a:02d} c{b:02d}' for a, b in zip(r+1, c+1)],
                                  row=(r+1).astype(np.uint16), col=(c+1).astype(np.uint16),
                                  rx=origin[0] + (c+.5)*side, ry=origin[1] + (r+.5)*side,
                                  n_cells=np.bincount(local).astype(np.uint32), n_umi=counts.sum(axis=1, dtype=np.int64),
                                  n_genes_detected=(counts > 0).sum(axis=1).astype(np.uint16)))
        for level in LEVELS:
            labels, codes = np.unique(part[level].to_numpy(), return_inverse=True)
            table[level], table[f'{level}_fraction'] = main_labels(codes, labels, local, n)
        nz_rows, nz_cols = np.nonzero(counts)
        indptr, gene_index, rank2, zero2 = rank_rows(nz_rows, nz_cols, counts[nz_rows, nz_cols], n, g)
        check = rng.choice(n, min(n, args.check_bins), replace=False)
        validated += validate_rows(indptr, gene_index, rank2, zero2, [(b, counts[b].astype(np.float64)) for b in check], g, rng)
        files = dict(counts=f'counts/{slice_id}.npz', ranks=f'ranks/{slice_id}.npz')
        np.savez_compressed(staging/files['counts'], counts=counts)
        # The shard format of the cell batches; its rows (cell_index) are bins.
        np.savez_compressed(staging/files['ranks'], cell_index=np.arange(offset, offset+n, dtype=np.uint32), indptr=indptr.astype(np.uint32),
                            gene_index=gene_index, rank2=rank2, zero_rank2=zero2)
        with np.load(staging/files['ranks'], allow_pickle=False) as z:  # round-trip check of the serialized ranks
            np.testing.assert_array_equal(z['gene_index'], gene_index)
            np.testing.assert_array_equal(z['rank2'], rank2)
        shutil.copy2(args.cells/entry['pseudobulk'], staging/entry['pseudobulk'])
        for name in (*files.values(), entry['pseudobulk']):
            record(staging/name)
        cell_bins.append(pd.DataFrame(dict(cell_index=part.cell_index.to_numpy(), bin_index=(offset + local).astype(np.uint32))))
        tables.append(table)
        slice_entries.append(dict(id=slice_id, n_cells=len(part), n_bins=n, bin_index_start=offset, bin_index_stop=offset+n, **files,
                                  pseudobulk=entry['pseudobulk'], source=entry['source'], source_bytes=entry['source_bytes'],
                                  source_sha256=entry['source_sha256'], rows=rows, umi=umi, seconds=round(time.perf_counter()-t, 1)))
        offset += n
        print(json.dumps(slice_entries[-1]), flush=True)

    bins = pd.concat(tables, ignore_index=True)
    bins.insert(0, 'bin_index', np.arange(len(bins), dtype=np.uint32))
    bins['draw_order'] = np.arange(len(bins), dtype=np.uint32)  # squares do not overlap
    bins.to_parquet(staging/'bins.parquet', index=False); record(staging/'bins.parquet')
    pd.concat(cell_bins, ignore_index=True).to_parquet(staging/'cell_bins.parquet', index=False); record(staging/'cell_bins.parquet')
    for name in ['genes.parquet', 'annotations.json']:
        shutil.copy2(args.cells/name, staging/name); record(staging/name)
    parents = {level: cells.drop_duplicates(level).set_index(level)[parent].to_dict() for level, parent in PARENT.items()}
    manifest = dict(
        format=source['format'], layout='square-bins-v2', dataset=out.name,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=dict(path=os.path.relpath(args.cells, out), dataset=source['dataset'], created_utc=source['created_utc']),
        n_bins=len(bins), n_cells=len(cells), n_genes=g, n_slices=len(slices), seed=args.seed,
        bin_um=bin_um, bin_side=side, origin_rxry=list(origin), lattice=dict(rows=n_rows, cols=n_cols),
        observations='square bins of cells; counts are the summed retained-gene UMIs of the cells whose centroid lies in the bin',
        binning='one lattice of squares (bin_side units) for all slices, from origin_rxry, the top left of their shared registered frame; '
                'rows from the top (dorsal), columns from the left, numbered from 1; only squares with cells are bins',
        coordinates=dict(columns='rx, ry', units='registered Stereo-seq spot units (500 nm), bin centre', um_per_unit=UM_PER_UNIT),
        main_labels="each bin's most common label per level among its cells (ties: first in sorted order); <level>_fraction is its share",
        ranks=dict(source['ranks'], rows='bins (cell_index holds bin_index)'),
        pseudobulk=dict(source['pseudobulk'], description='label pseudobulks of the cell dataset, copied for these slices: ' + source['pseudobulk']['description']),
        parents=parents,
        validation=dict(cell_umi_totals_match=len(cells), gene_totals_match_cell_class_pseudobulk=len(slices),
                        exact_rank_and_pairwise_auc_rows=validated, ranks_roundtripped=True),
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__, scipy=scipy.__version__),
        slices=slice_entries, files=manifest_files, seconds=round(time.perf_counter()-started, 1))
    (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['files', 'slices', 'parents', 'pseudobulk']}, indent=2))
    print('OUTPUT_BYTES', sum(p.stat().st_size for p in out.rglob('*') if p.is_file()), flush=True)


if __name__ == '__main__':
    main()
