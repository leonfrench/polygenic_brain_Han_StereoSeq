"""Square bins of cells, the cell app's binned colour mode (spatial_app.bins)."""
import json
import numpy as np
import pandas as pd
import pytest
from spatial_app.bins import MIN_BIN_CELLS, load_bins, bin_scores
from spatial_app.scoring import resolve_query
from test_scoring import pack, pairwise

ANNOTATIONS = {'cell_subclass': dict(kind='cell_type', title='Cell subclass', labels={'A': '#FF0000'}, descriptions={}),
               'region_area': dict(kind='region', title='Brain area', labels={'MH': '#FF909F'}, descriptions={'MH': 'Medial habenula'})}


def test_bins_score_slice_by_slice_in_drawing_order_and_skip_shallow_bins(tmp_path):
    counts = np.array([[0, 3, 1, 0, 2, 0], [5, 0, 0, 1, 0, 0], [1, 1, 4, 4, 0, 2]])  # bins 0 and 1 in S1, bin 2 in S2
    (tmp_path/'ranks').mkdir()
    for name, rows, start in [('S1', counts[:2], 0), ('S2', counts[2:], 2)]:
        shard = pack(rows)
        shard['cell_index'] = np.arange(start, start+len(rows), dtype=np.uint32)  # global bin indices
        np.savez(tmp_path/'ranks'/f'{name}.npz', **shard)
    slices = [dict(id='S1', ranks='ranks/S1.npz'), dict(id='S2', ranks='ranks/S2.npz')]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(layout='square-bins-v2', n_bins=3, slices=slices)))
    pd.DataFrame(dict(bin_index=[0, 1, 2], bin=['S1 r01 c01', 'S1 r01 c02', 'S2 r01 c01'], n_cells=[MIN_BIN_CELLS, 1, 40],
                      draw_order=[2, 0, 1])).to_parquet(tmp_path/'bins.parquet')
    manifest, bins, levels = load_bins(tmp_path, ANNOTATIONS)
    assert bins.bin_index.tolist() == [1, 2, 0]  # drawing order
    assert [levels[k]['title'] for k in ('cell_subclass', 'region_area', 'bin')] == ['Main cell subclass', 'Main brain area', 'Bin']
    assert levels['bin']['descriptions'] == {'S1 r01 c01': f'{MIN_BIN_CELLS} cells', 'S1 r01 c02': '1 cell, too few to score',
                                             'S2 r01 c01': '40 cells'}
    symbols = [f'G{i}' for i in range(6)]
    for background in ('', 'G0 G1 G2 G4'):  # all genes, then ranks recomputed inside a custom universe
        query = resolve_query('G1 G2', background, symbols)
        universe = np.arange(6) if query.background is None else np.asarray(query.background)
        expected = pairwise(counts, np.asarray(query.target), universe)[bins.bin_index.to_numpy()]
        expected[bins.n_cells.to_numpy() < MIN_BIN_CELLS] = np.nan
        np.testing.assert_allclose(bin_scores(tmp_path, manifest, bins, query, len(symbols)), expected)


def test_only_per_slice_rank_layouts_are_offered(tmp_path):
    pd.DataFrame(dict(bin_index=[0], bin=['T319 r01 c01'], n_cells=[30], draw_order=[0])).to_parquet(tmp_path/'bins.parquet')
    # The first T319 build (square-bins-v1) kept one ranks.npz for all slices, so its slice entries have no ranks.
    for manifest in (dict(layout='square-bins-v1', n_bins=1, slices=[dict(id='T319')]),
                     dict(layout='square-bins-v2', n_bins=1, slices=[dict(id='T319')])):
        (tmp_path/'manifest.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match='Unsupported bins layout'):
            load_bins(tmp_path, ANNOTATIONS)
