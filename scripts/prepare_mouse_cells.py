"""Stereo-seq cell-segmented mouse brain slices -> exact sparse ranks in per-slice batches.

Inputs (Han et al., mouse 1) in --raw:
  total_gene_<SLICE>_mouse_f001_2D_<MOUSE>-20230119.txt.gz  one row per spot x gene:
      gene, x, y, umi_count, cell_label (0 = no cell), gene_area (region code), rx, ry
  stereoseq.celltypeTransfer.2mice.all.tsv.gz  mouse, section_id, cell_id, cluster/subclass/class, color
  regions-<MOUSE>-20230119.rmDup.tsv  gene_area -> L/R-prefixed CCF acronym path and description

Each cell's counts are the summed UMIs of its spots. Only cells with a transferred cell
type are kept. The cell's region is the gene_area code holding most of its UMIs.
All processed slices share one gene universe so their AUROCs are directly comparable;
rerun with the full slice list when slices are added or removed.

  .conda-env/bin/python scripts/prepare_mouse_cells.py --slices T258 T299 T335
"""
from pathlib import Path
import argparse, colorsys, datetime, hashlib, json, platform, re, shutil, subprocess, time
import numpy as np
import pandas as pd
import polars as pl
import scipy
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
MGI_URLS = {'MRK_List2.rpt': 'https://www.informatics.jax.org/downloads/reports/MRK_List2.rpt',
            'MRK_List1.rpt': 'https://www.informatics.jax.org/downloads/reports/MRK_List1.rpt'}
ALLEN_URL = 'http://api.brain-map.org/api/v2/structure_graph_download/1.json'
SCHEMA = {'gene': pl.Utf8, 'x': pl.Int32, 'y': pl.Int32, 'umi_count': pl.Int32,
          'cell_label': pl.Int64, 'gene_area': pl.Int64, 'rx': pl.Float64, 'ry': pl.Float64}
# The region table's author notes are in Chinese; these are literal translations.
NOTES = [(r'未能区分CA1、CA2、CA3，故将其合并到CA子级', 'CA1, CA2 and CA3 not distinguished; merged into CA'),
         (r'未能区分(L\w+)、(L\w+)', r'\1 and \2 not distinguished'),
         (r'未能区分(\w+)亚区', r'\1 subareas not distinguished'),
         (r'Allen 未区分此区', 'not subdivided in the Allen reference'),
         (r'背景', 'background')]


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(url, path):
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.part')
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def slice_number(name):
    return int(re.sub(r'\D', '', name) or 0)


# ---------------------------------------------------------------- gene universe

def mgi_tables(reference):
    current = pd.read_csv(download(MGI_URLS['MRK_List2.rpt'], reference/'MRK_List2.rpt'), sep='\t', dtype=str, keep_default_na=False)
    current = current[current['Status'].eq('O') & current['Marker Type'].eq('Gene')]
    pc = current[current['Feature Type'].eq('protein coding gene')][['MGI Accession ID', 'Marker Symbol', 'Marker Name']]
    pc.columns = ['mgi_id', 'symbol', 'name']
    full = pd.read_csv(download(MGI_URLS['MRK_List1.rpt'], reference/'MRK_List1.rpt'), sep='\t', dtype=str, keep_default_na=False)
    withdrawn = full[full['Status'].eq('W')][['Marker Symbol', 'Current MGI Accession ID (if withdrawn)']]
    withdrawn.columns = ['old_symbol', 'mgi_id']
    return current, pc.reset_index(drop=True), withdrawn


def map_genes(names, current, pc, withdrawn):
    """Current MGI symbol first, then an unambiguous withdrawn symbol. No synonym guessing."""
    by_symbol = dict(zip(current['Marker Symbol'], current['MGI Accession ID']))
    pc_ids = dict(zip(pc.mgi_id, pc.symbol))
    old = withdrawn[withdrawn.mgi_id.ne('')].groupby('old_symbol').mgi_id.agg(lambda s: sorted(set(s)))
    rows = []
    for name in names:
        ident, method = by_symbol.get(name), 'current_symbol'
        if ident is None:
            candidates = old.get(name, [])
            method = 'withdrawn_symbol' if candidates else 'unmapped'
            ident = candidates[0] if len(candidates) == 1 else None
            if len(candidates) > 1:
                method = 'ambiguous_withdrawn_symbol'
        status = 'retained' if ident in pc_ids else ('not_protein_coding' if ident else method)
        rows.append((name, ident, pc_ids.get(ident), method, status))
    return pd.DataFrame(rows, columns=['source_gene', 'mgi_id', 'symbol', 'mapping_method', 'status'])


# ---------------------------------------------------------------- annotations

def allen_structures(reference):
    graph = json.loads(download(ALLEN_URL, reference/'allen_structure_graph_1.json').read_text())
    table = {}
    stack = [graph['msg'][0]]
    while stack:
        node = stack.pop()
        table[node['acronym']] = dict(name=node['name'], color='#'+node['color_hex_triplet'].upper())
        stack.extend(node['children'])
    return table


def region_table(path, allen):
    regions = pd.read_csv(path, sep='\t', dtype={'gene_area': np.int64, 'area_name': str, 'description': str})
    if regions.gene_area.duplicated().any():
        raise ValueError('Region codes must be unique')
    rows = []
    for code, name, note in regions.itertuples(index=False, name=None):
        if code == 0:
            rows.append((code, '', 'background', 'background', 'background'))
            continue
        hemisphere, parts = name[0], name[1:].split('-')
        if hemisphere not in 'LR' or not parts[0]:
            raise ValueError(f'Unexpected region name {name!r}')
        rows.append((code, hemisphere, parts[0], parts[min(1, len(parts)-1)], parts[-1]))
    out = pd.DataFrame(rows, columns=['gene_area', 'hemisphere', 'region_division', 'region_area', 'region'])
    out['path'] = regions.area_name.str.replace(r'^[LR]', '', regex=True).where(regions.gene_area.ne(0), 'background').to_numpy()
    descriptions = {}
    for path, note in zip(out.path, regions.description):
        english = note
        for pattern, text in NOTES:
            english = re.sub(pattern, text, english)
        english = re.sub(r'^\w+:\s*', '', english) if english != note else english
        if english != note:
            # Translated notes describe a merged region, so name its Allen parent.
            parent = next((allen[p]['name'] for p in path.split('-')[-2::-1] if p in allen), None)
            english = f'{parent}; {english}' if parent else english
        descriptions.setdefault(path.split('-')[-1], english)
    return out, descriptions


def hex_rgb(color):
    return tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))


def separate_colors(colors):
    """Spread lightness inside groups of (near-)identical colours, keeping each hue family."""
    labels = sorted(colors)
    groups, assigned = [], {}
    for label in labels:
        rgb = np.array(hex_rgb(colors[label]))
        for group in groups:
            if np.abs(rgb-group['rgb']).max() < 12/255:
                group['labels'].append(label); break
        else:
            groups.append(dict(rgb=rgb, labels=[label]))
    for group in groups:
        m = len(group['labels'])
        for i, label in enumerate(group['labels']):
            h, l, s = colorsys.rgb_to_hls(*hex_rgb(colors[label]))
            if m > 1:
                l = float(np.clip(l + np.linspace(-1, 1, m)[i]*min(0.22, 0.09*(m-1)), 0.12, 0.88))
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            assigned[label] = '#%02X%02X%02X' % tuple(round(v*255) for v in (r, g, b))
    return assigned


def annotation_levels(cells, types, regions, descriptions, allen):
    # Colours come from the whole transfer table, so they do not change with the slice list.
    levels = {}
    types = types.select('cell_cluster', 'cell_subclass', 'cell_class', 'color').to_pandas()
    cluster_color = types.groupby('cell_cluster').color.agg(lambda s: s.mode().iloc[0])
    counts = types.groupby('cell_cluster').size()
    for column, title in [('cell_class', 'Cell class'), ('cell_subclass', 'Cell subclass'), ('cell_cluster', 'Cell cluster')]:
        if column == 'cell_cluster':
            colors = cluster_color.str.upper().to_dict()
        else:
            # A derived level takes the colour of its most numerous cluster.
            parent = types.groupby('cell_cluster')[column].first()
            frame = pd.DataFrame(dict(parent=parent, n=counts, color=cluster_color))
            colors = separate_colors(frame.sort_values('n', ascending=False).groupby('parent').color.first().str.upper().to_dict())
        present = set(cells[column].astype(str))
        levels[column] = dict(kind='cell_type', title=title, labels={k: v for k, v in sorted(colors.items()) if k in present}, descriptions={})
    for column, title in [('region_division', 'Brain division'), ('region_area', 'Brain area'), ('region', 'Finest region')]:
        colors, desc = {}, {}
        for label, path in regions.drop_duplicates(column).set_index(column)['path'].items():
            if label == 'background':
                colors[label], desc[label] = '#C9CED6', 'Background or unannotated tissue'
                continue
            tokens = path.split('-')[:{'region_division': 1, 'region_area': 2, 'region': len(path.split('-'))}[column]]
            colors[label] = next((allen[t]['color'] for t in tokens[::-1] if t in allen), '#B0B0B0')
            desc[label] = descriptions.get(label) or allen.get(label, {}).get('name', label)
        present = set(cells[column].astype(str))
        colors = separate_colors({k: v for k, v in colors.items() if k in present})
        levels[column] = dict(kind='region', title=title, labels=colors, descriptions={k: desc[k] for k in colors})
    return levels


# ---------------------------------------------------------------- per-slice processing

def read_slice(path, mouse, slice_id, types, regions, work):
    started = time.perf_counter()
    # Large slices reach ~7.5 GB of text. Decompress to disk and aggregate with the
    # streaming engine, so neither the text nor the full table is held in memory.
    text = work/f'{slice_id}.tsv'
    with open(text, 'wb') as out:
        subprocess.run(['gzip', '-dc', str(path)], stdout=out, check=True)
    rows = pl.scan_csv(text, separator='\t', schema=SCHEMA).with_columns(pl.col('umi_count').cast(pl.Int64))
    in_cells = rows.filter(pl.col('cell_label') != 0)
    totals, geometry, area, counts = pl.collect_all([
        rows.select(pl.len().alias('rows'), pl.col('umi_count').sum().alias('umi')),
        # Centroid of the cell's unique spots; rx, ry are a fixed transform of x, y.
        in_cells.group_by('cell_label', 'x', 'y').agg(pl.col('rx').first(), pl.col('ry').first())
                .group_by('cell_label').agg(pl.len().alias('n_spots'), pl.col('x').mean(), pl.col('y').mean(),
                                            pl.col('rx').mean(), pl.col('ry').mean()),
        in_cells.group_by('cell_label', 'gene_area').agg(pl.col('umi_count').sum()),
        in_cells.group_by('cell_label', 'gene').agg(pl.col('umi_count').sum())], engine='streaming')
    text.unlink()
    total_rows, total_umi = int(totals['rows'][0]), int(totals['umi'][0])
    area = area.sort(['cell_label', 'umi_count', 'gene_area'], descending=[False, True, False])
    umi = area.group_by('cell_label').agg(pl.col('umi_count').sum().alias('n_umi'))
    major = area.group_by('cell_label', maintain_order=True).first().rename({'umi_count': 'region_umi'})
    section = types.filter((pl.col('mouse') == mouse) & (pl.col('section_id') == slice_id)).drop('mouse', 'section_id')
    cells = (geometry.join(umi, on='cell_label').join(major, on='cell_label')
             .join(section, left_on='cell_label', right_on='cell_id', how='left').sort('cell_label'))
    segmented = cells.height
    cells = cells.filter(pl.col('cell_cluster').is_not_null())
    missing_types = section.height - section.join(cells.select('cell_label'), left_on='cell_id', right_on='cell_label', how='semi').height
    unknown = set(cells['gene_area'].unique().to_list()) - set(regions.gene_area)
    if unknown:
        raise ValueError(f'{slice_id}: region codes absent from the region table: {sorted(unknown)[:10]}')
    counts = counts.join(cells.select('cell_label'), on='cell_label', how='semi')
    cells = cells.with_columns((pl.col('region_umi')/pl.col('n_umi')).alias('region_fraction')).drop('region_umi')
    report = dict(slice=slice_id, source=path.name, source_bytes=path.stat().st_size, source_sha256=sha256(path),
                  rows=total_rows, umi=total_umi, segmented_cells=segmented, typed_cells=cells.height,
                  untyped_segmented_cells=segmented-cells.height, typed_cells_without_segmentation=missing_types,
                  read_seconds=round(time.perf_counter()-started, 2))
    print(json.dumps(report), flush=True)
    return cells.to_pandas(), counts, report


def rank_rows(rows, cols, values, n_rows, n_genes):
    """Doubled average ranks per row; positive values only, zeros share (Z+1)/2."""
    rows, cols, values = (np.asarray(a, dtype=np.int64) for a in (rows, cols, values))
    # One integer key per sort; np.lexsort is much slower on unordered rows.
    order = np.argsort(rows*(int(values.max(initial=0))+1) + values)
    rows, cols, values = rows[order], cols[order], values[order]
    counts = np.bincount(rows, minlength=n_rows)
    zeros = n_genes - counts
    start = np.r_[0, np.flatnonzero((np.diff(rows) != 0) | (np.diff(values) != 0)) + 1]
    end = np.r_[start[1:], len(rows)]
    row_start = np.r_[0, np.cumsum(counts)[:-1]]
    # A 0-based [start,end) tie inside the positive block has doubled 1-based rank start+end+1.
    rank2 = np.repeat(start + end + 1 - 2*row_start[rows[start]] + 2*zeros[rows[start]], end - start)
    order = np.argsort(rows*n_genes + cols)
    indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    return indptr, cols[order].astype(np.uint16), rank2[order].astype(np.uint16), np.where(zeros > 0, zeros+1, 0).astype(np.uint16)


LEVELS = ['cell_class', 'cell_subclass', 'cell_cluster', 'region_division', 'region_area', 'region']
# The right hemisphere is annotated mostly as whole structures (e.g. one 'Isocortex' label),
# so region pseudobulks use left-hemisphere cells only. Cell types do not depend on it.
PSEUDOBULK_CELLS = {level: 'left hemisphere' if level.startswith('region') else 'all' for level in LEVELS}


def pseudobulk(rows, cols, values, codes, n_genes, include=None):
    """Summed UMIs per (label, gene) for the labels present among the included cells.

    rows index cells; codes give each cell's label; include (optional) selects cells.
    """
    include = np.ones(len(codes), bool) if include is None else np.asarray(include, bool)
    present, inverse = np.unique(codes[include], return_inverse=True)
    local = np.full(len(codes), -1); local[include] = inverse
    keep = include[rows]
    sums = np.bincount(local[rows[keep]]*n_genes + cols[keep], weights=values[keep], minlength=len(present)*n_genes)
    if sums.max(initial=0) >= 2**32:
        raise ValueError('Pseudobulk counts exceed uint32')
    return present, np.bincount(inverse, minlength=len(present)), sums.reshape(len(present), n_genes).astype(np.uint32)


def validate_rows(indptr, cols, rank2, zero2, dense_rows, n_genes, rng):
    checked = 0
    for row, dense in dense_rows:
        a, b = indptr[row], indptr[row+1]
        decoded = np.full(n_genes, zero2[row]/2); decoded[cols[a:b]] = rank2[a:b]/2
        np.testing.assert_array_equal(decoded, rankdata(dense, method='average'))
        selected = rng.choice(n_genes, 100, replace=False)
        other = np.setdiff1d(np.arange(n_genes), selected)
        truth = np.mean((dense[selected, None] > dense[other]) + 0.5*(dense[selected, None] == dense[other]))
        auc = (decoded[selected].sum() - 100*101/2) / (100*(n_genes-100))
        np.testing.assert_allclose(auc, truth, atol=1e-12)
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/Han et al.')
    parser.add_argument('--slices', nargs='*', help='Slice IDs, e.g. T258 T299. Default: every total_gene_* file in --raw.')
    parser.add_argument('--mouse', default='mouse1')
    parser.add_argument('--out', type=Path, default=ROOT/'data/mouse1')
    parser.add_argument('--reference', type=Path, default=ROOT/'data/reference')
    parser.add_argument('--batch-cells', type=int, default=16384, help='Maximum cells per batch.')
    parser.add_argument('--batch-entries', type=int, default=12_000_000,
                        help='Maximum stored (cell, detected gene) entries per batch; keeps files well under 95 MB.')
    parser.add_argument('--min-cells', type=int, default=0, help='Drop genes detected in fewer cells (all slices together).')
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--force', action='store_true', help='Replace an existing output directory.')
    args = parser.parse_args()
    started = time.perf_counter()
    files = {re.match(r'total_gene_(\w+?)_mouse', p.name).group(1): p
             for p in args.raw.glob(f'total_gene_*_{args.mouse}-*.txt.gz')}
    slices = sorted(args.slices or files, key=slice_number)
    if not slices or any(s not in files for s in slices):
        raise SystemExit(f'Missing slice files for {[s for s in slices if s not in files] or "any slice"} in {args.raw}')
    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        raise SystemExit(f'{args.out} is not empty; pass --force to replace it')
    staging = args.out.with_name(args.out.name + '.staging')
    if staging.exists():
        shutil.rmtree(staging)
    (staging/'progressive').mkdir(parents=True)
    work = staging/'.work'; work.mkdir()

    allen = allen_structures(args.reference)
    current, pc, withdrawn = mgi_tables(args.reference)
    type_path = args.raw/'stereoseq.celltypeTransfer.2mice.all.tsv.gz'
    region_path = next(args.raw.glob(f'regions-{args.mouse}-*.tsv'))
    types = pl.read_csv(type_path, separator='\t', schema_overrides={'cell_id': pl.Int64})
    regions, descriptions = region_table(region_path, allen)

    # Pass 1: aggregate each slice to cell x source-gene UMIs (kept on disk).
    metas, reports, source_genes = [], [], set()
    for slice_id in slices:
        cells, counts, report = read_slice(files[slice_id], args.mouse, slice_id, types, regions, work)
        counts.write_parquet(work/f'{slice_id}.parquet')
        source_genes |= set(counts['gene'].unique().to_list())
        cells.insert(0, 'slice', slice_id)
        metas.append(cells); reports.append(report)
        del counts
    cells = pd.concat(metas, ignore_index=True)
    mapping = map_genes(sorted(source_genes), current, pc, withdrawn)

    # Universe = protein-coding genes detected in any typed cell of the processed slices.
    kept = mapping[mapping.status.eq('retained')]
    symbols = np.array(sorted(kept.symbol.unique()))
    gene_lookup = pl.DataFrame({'gene': kept.source_gene.to_numpy(),
                                'gene_index': pd.Index(symbols).get_indexer(kept.symbol).astype(np.int32)})
    n = len(cells)
    cells.insert(0, 'cell_index', np.arange(n, dtype=np.uint32))
    offsets = dict(zip(slices, np.r_[0, np.cumsum([len(m) for m in metas])[:-1]]))
    detection = np.zeros(len(symbols), np.int64); totals = np.zeros(len(symbols), np.float64)
    retained_umi = np.zeros(n, np.int64); detected = np.zeros(n, np.int64)
    aggregated, per_slice = {}, {}
    for slice_id, meta in zip(slices, metas):
        local = pl.DataFrame({'cell_label': meta.cell_label.to_numpy(), 'row': np.arange(len(meta), dtype=np.int64)})
        counts = (pl.read_parquet(work/f'{slice_id}.parquet').join(gene_lookup, on='gene')
                    .group_by('cell_label', 'gene_index').agg(pl.col('umi_count').sum())  # duplicate symbols summed before ranking
                    .join(local, on='cell_label'))
        rows, cols, values = (counts['row'].to_numpy(), counts['gene_index'].to_numpy(), counts['umi_count'].to_numpy())
        per_slice[slice_id] = np.bincount(cols, minlength=len(symbols))
        detection += per_slice[slice_id]; totals += np.bincount(cols, weights=values, minlength=len(symbols))
        retained_umi[offsets[slice_id]:offsets[slice_id]+len(meta)] = np.bincount(rows, weights=values, minlength=len(meta))
        detected[offsets[slice_id]:offsets[slice_id]+len(meta)] = np.bincount(rows, minlength=len(meta))
        aggregated[slice_id] = work/f'{slice_id}.ranked.parquet'
        counts.select('row', 'gene_index', 'umi_count').write_parquet(aggregated[slice_id])
        (work/f'{slice_id}.parquet').unlink()
    keep = detection >= args.min_cells
    remap = np.full(len(symbols), -1, np.int64); remap[keep] = np.arange(keep.sum())
    symbols = symbols[keep]; g = len(symbols)
    if not 1 < g < 32768:
        raise ValueError('uint16 doubled ranks require 2 <= genes <= 32767')
    mapping.loc[mapping.status.eq('retained') & ~mapping.symbol.isin(symbols), 'status'] = 'below_min_cells'

    genes = pc.set_index('symbol').loc[symbols].reset_index()[['symbol', 'mgi_id', 'name']]
    genes.insert(0, 'gene_index', np.arange(g, dtype=np.uint16))
    genes['n_cells_detected'] = detection[keep]; genes['total_counts'] = totals[keep]
    genes['source_genes'] = kept.groupby('symbol').source_gene.agg('|'.join).reindex(symbols).to_numpy()

    region_columns = regions.drop(columns='path')
    cells = cells.merge(region_columns, on='gene_area', how='left', validate='many_to_one')
    cells = cells.rename(columns={'cell_label': 'cell_id', 'gene_area': 'region_code', 'color': 'cluster_color'})
    cells.insert(3, 'source_obs_id', args.mouse + '_' + cells.slice + '_' + cells.cell_id.astype(str))
    cells['n_umi_retained'] = retained_umi; cells['n_genes_detected'] = detected
    # Random global drawing order, so overlapping points do not favour any slice or type.
    cells['draw_order'] = np.random.default_rng(args.seed).permutation(n).astype(np.uint32)
    cells['batch'] = ''; cells['batch_row'] = np.zeros(n, np.int32)
    for column in ['slice', 'cell_cluster', 'cell_subclass', 'cell_class', 'hemisphere', 'region_division', 'region_area', 'region']:
        cells[column] = cells[column].astype(str)

    # Pass 2: exact ranks, randomly ordered per-slice batches and per-slice pseudobulk counts.
    labels = {level: np.unique(cells[level].to_numpy()) for level in LEVELS}
    (staging/'pseudobulk').mkdir()
    rng = np.random.default_rng(args.seed)
    manifest_files, slice_entries, validated = [], [], 0
    def record(path, **extra):
        item = dict(path=str(path.relative_to(staging)), bytes=path.stat().st_size, sha256=sha256(path), **extra)
        if item['bytes'] >= 95_000_000:
            raise ValueError(f'File exceeds conservative GitHub limit: {path}')
        manifest_files.append(item); return item
    for slice_id, meta in zip(slices, metas):
        t = time.perf_counter()
        counts = pl.read_parquet(aggregated[slice_id])
        rows = counts['row'].to_numpy(); cols = remap[counts['gene_index'].to_numpy()]; values = counts['umi_count'].to_numpy()
        if args.min_cells:
            rows, values, cols = rows[cols >= 0], values[cols >= 0], cols[cols >= 0]
        m = len(meta)
        indptr, gene_index, rank2, zero2 = rank_rows(rows, cols, values, m, g)
        check = rng.choice(m, min(m, 32), replace=False)
        dense_rows = []
        for row in check:
            dense = np.zeros(g); hit = rows == row
            dense[cols[hit]] = values[hit]; dense_rows.append((row, dense))
        validated += validate_rows(indptr, gene_index, rank2, zero2, dense_rows, g, rng)
        order = rng.permutation(m)
        # Equal-size batches, small enough for both the cell and stored-entry limits.
        size = max(1, min(args.batch_cells, int(args.batch_entries*m/max(1, len(gene_index)))))
        size = -(-m // -(-m // size))
        batches = []
        for b, start in enumerate(range(0, m, size)):
            local = order[start:start+size]
            lengths = indptr[local+1] - indptr[local]
            ptr = np.r_[0, np.cumsum(lengths)]
            gather = np.repeat(indptr[local] - ptr[:-1], lengths) + np.arange(ptr[-1])
            path = staging/'progressive'/slice_id/f'{b:05d}.npz'; path.parent.mkdir(exist_ok=True)
            global_rows = (offsets[slice_id] + local).astype(np.uint32)
            np.savez_compressed(path, cell_index=global_rows, indptr=ptr.astype(np.uint32),
                                gene_index=gene_index[gather], rank2=rank2[gather], zero_rank2=zero2[local])
            with np.load(path, allow_pickle=False) as z:  # round-trip check of the serialized batch
                np.testing.assert_array_equal(z['gene_index'], gene_index[gather])
                np.testing.assert_array_equal(z['rank2'], rank2[gather])
            cells.loc[global_rows, 'batch'] = str(path.relative_to(staging))
            cells.loc[global_rows, 'batch_row'] = np.arange(len(local), dtype=np.int32)
            batches.append(record(path, slice=slice_id, n_cells=len(local))['path'])
        part = cells.iloc[offsets[slice_id]:offsets[slice_id]+m]
        arrays = {}
        for level in LEVELS:
            codes = np.searchsorted(labels[level], part[level].to_numpy())
            include = part.hemisphere.eq('L').to_numpy() if PSEUDOBULK_CELLS[level] == 'left hemisphere' else np.ones(m, bool)
            present, n_cells, sums = pseudobulk(rows, cols, values, codes, g, include)
            np.testing.assert_array_equal(sums.sum(axis=0, dtype=np.float64),
                                          np.bincount(cols, weights=values*include[rows], minlength=g))
            arrays.update({f'{level}_index': present.astype(np.uint16), f'{level}_cells': n_cells.astype(np.uint32),
                           f'{level}_counts': sums})
        path = staging/'pseudobulk'/f'{slice_id}.npz'
        np.savez_compressed(path, **arrays)
        record(path, slice=slice_id, layout='pseudobulk')
        report = next(r for r in reports if r['slice'] == slice_id)
        slice_entries.append(dict(id=slice_id, n_cells=m, cell_index_start=int(offsets[slice_id]), cell_index_stop=int(offsets[slice_id]+m),
                                  bbox_rxry=[float(part.rx.min()), float(part.ry.min()), float(part.rx.max()), float(part.ry.max())],
                                  batches=batches, pseudobulk=str(path.relative_to(staging)),
                                  rank_seconds=round(time.perf_counter()-t, 2), **report))
        print(f'{slice_id}: {m:,} cells ranked into {len(batches)} batches in {time.perf_counter()-t:.1f}s', flush=True)

    shutil.rmtree(work)
    levels = annotation_levels(cells, types, regions, descriptions, allen)
    (staging/'annotations.json').write_text(json.dumps(levels, indent=1, ensure_ascii=False)+'\n'); record(staging/'annotations.json')
    cells.to_parquet(staging/'cells.parquet', index=False); record(staging/'cells.parquet')
    genes.to_parquet(staging/'genes.parquet', index=False); record(staging/'genes.parquet')
    mapping.to_parquet(staging/'gene_mapping.parquet', index=False); record(staging/'gene_mapping.parquet')
    detection_by_slice = pd.DataFrame({'gene_index': np.tile(np.arange(g, dtype=np.uint16), len(slices)),
                                       'slice': np.repeat(slices, g),
                                       'n_cells_detected': np.concatenate([per_slice[s][keep] for s in slices])})
    detection_by_slice.to_parquet(staging/'gene_detection.parquet', index=False); record(staging/'gene_detection.parquet')
    reference = args.reference
    manifest = dict(
        format='sparse-doubled-average-ranks-v1', layout='per-slice-random-batches-v1', dataset=args.mouse,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        n_cells=n, n_genes=g, n_slices=len(slices), seed=args.seed, min_cells=args.min_cells, batch_cells=args.batch_cells, batch_entries=args.batch_entries,
        observations='segmented cells with a transferred cell type; counts are summed spot UMIs',
        coordinates=dict(columns='rx, ry', units='registered Stereo-seq spot units (500 nm), centroid of the cell\'s spots',
                         raw='x, y are the unregistered chip coordinates'),
        region_assignment='gene_area code with the largest UMI share in the cell; 0 = background',
        sources=dict(cell_types=dict(path=type_path.name, sha256=sha256(type_path)),
                     regions=dict(path=region_path.name, sha256=sha256(region_path)),
                     mgi={k: dict(url=v, sha256=sha256(reference/k)) for k, v in MGI_URLS.items()},
                     allen=dict(url=ALLEN_URL, sha256=sha256(reference/'allen_structure_graph_1.json'))),
        gene_mapping_counts=mapping.status.value_counts().to_dict(),
        retained_mapping_methods=mapping.loc[mapping.status.eq('retained'), 'mapping_method'].value_counts().to_dict(),
        ranks=dict(direction='ascending: larger rank means higher expression', ties='average', scale=2,
                   zeros='all unstored genes have rank zero_rank2 / 2; stored ranks are absolute',
                   universe='current MGI protein-coding genes detected in at least min_cells typed cells of the processed slices',
                   duplicate_genes='sum counts before ranking', auroc='(sum(rank of set) - k*(k+1)/2) / (k*(G-k))'),
        annotation_levels={k: dict(kind=v['kind'], title=v['title'], n_labels=len(v['labels'])) for k, v in levels.items()},
        pseudobulk=dict(description='per slice and annotation level: summed retained-gene UMIs per (label, gene), '
                                    '<level>_counts rows follow <level>_index into labels[level]; <level>_cells counts cells',
                        labels={level: labels[level].tolist() for level in LEVELS}, cells=PSEUDOBULK_CELLS),
        validation=dict(exact_rank_and_pairwise_auc_rows=validated, batches_roundtripped=True),
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      polars=pl.__version__, scipy=scipy.__version__),
        slices=slice_entries, files=manifest_files, seconds=round(time.perf_counter()-started, 1))
    (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if args.out.exists():
        shutil.rmtree(args.out)
    staging.rename(args.out)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['files', 'slices']}, indent=2))
    print('OUTPUT_BYTES', sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file()), flush=True)


if __name__ == '__main__':
    main()
