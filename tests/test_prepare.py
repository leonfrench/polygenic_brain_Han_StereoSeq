import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

# Preprocessing needs requirements-prep.txt; an app-only install (requirements.txt) skips these tests.
pytest.importorskip('polars')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_mouse_cells import rank_rows, separate_colors, map_genes, region_table, pseudobulk  # noqa: E402


def test_rank_rows_matches_rankdata_with_ties_zeros_and_empty_rows():
    rng = np.random.default_rng(3)
    g, n = 40, 25
    dense = rng.integers(0, 4, (n, g)) * (rng.random((n, g)) < .5)
    dense[0] = 0                    # no detected genes
    dense[1] = rng.integers(1, 3, g)  # no zeros
    rows, cols = np.nonzero(dense)
    shuffle = rng.permutation(len(rows))  # aggregation output is unordered
    indptr, gene_index, rank2, zero2 = rank_rows(rows[shuffle], cols[shuffle], dense[rows, cols][shuffle], n, g)
    for row in range(n):
        decoded = np.full(g, zero2[row]/2)
        a, b = indptr[row], indptr[row+1]
        assert (np.diff(gene_index[a:b].astype(int)) > 0).all()
        decoded[gene_index[a:b]] = rank2[a:b]/2
        np.testing.assert_array_equal(decoded, rankdata(dense[row], method='average'))
    assert zero2[1] == 0 and zero2[0] == g+1


def test_separate_colors_splits_identical_colours_only():
    colors = separate_colors({'a': '#7A944D', 'b': '#7C9650', 'c': '#5858E0'})
    assert colors['c'] == '#5858E0'
    assert colors['a'] != colors['b']


def test_gene_mapping_prefers_current_symbols_and_rejects_ambiguous_withdrawn():
    current = pd.DataFrame({'Marker Symbol': ['Mbp', 'Gm1'], 'MGI Accession ID': ['MGI:1', 'MGI:2']})
    pc = pd.DataFrame({'mgi_id': ['MGI:1', 'MGI:3', 'MGI:4'], 'symbol': ['Mbp', 'Pakap', 'Other'], 'name': ['', '', '']})
    withdrawn = pd.DataFrame({'old_symbol': ['Akap2', 'Split', 'Split'], 'mgi_id': ['MGI:3', 'MGI:3', 'MGI:4']})
    table = map_genes(['Mbp', 'Gm1', 'Akap2', 'Split', 'Nope'], current, pc, withdrawn).set_index('source_gene')
    assert table.loc['Mbp', 'symbol'] == 'Mbp' and table.loc['Akap2', 'symbol'] == 'Pakap'
    assert table.loc['Gm1', 'status'] == 'not_protein_coding'
    assert table.loc['Split', 'status'] == 'ambiguous_withdrawn_symbol'
    assert table.loc['Nope', 'status'] == 'unmapped'


def test_region_names_split_hemisphere_and_translate_notes(tmp_path):
    path = tmp_path/'regions.tsv'
    path.write_text('gene_area\tarea_name\tdescription\n0\tbackground\t背景\n212\tLOLF-MOB-MOBgl\tMain olfactory bulb, glomerular layer\n'
                    '-507\tROLF-MOB\tMain olfactory bulb\n-1097\tRHY\tHypothalamus\n'
                    '2147483629\tLIsocortex-GU-GU6\tGU6: 未能区分L6a、L6b\n')
    allen = {'GU': dict(name='Gustatory areas', color='#009C75')}
    regions, descriptions = region_table(path, allen)
    regions = regions.set_index('gene_area')
    assert regions.loc[212, ['hemisphere', 'region_division', 'region_area', 'region']].tolist() == ['L', 'OLF', 'MOB', 'MOBgl']
    assert regions.loc[-507, ['hemisphere', 'region']].tolist() == ['R', 'MOB']
    assert regions.loc[-1097, ['region_division', 'region_area', 'region']].tolist() == ['HY', 'HY', 'HY']
    assert regions.loc[0, 'region'] == 'background'
    assert descriptions['GU6'] == 'Gustatory areas; L6a and L6b not distinguished'


def test_pseudobulk_sums_umis_per_present_label():
    rng = np.random.default_rng(5)
    dense = rng.poisson(1.5, (12, 7))
    codes = np.array([4, 4, 1, 4, 9, 1, 1, 4, 9, 9, 4, 1])  # global label indices; 0, 2, 3 ... absent
    rows, cols = np.nonzero(dense)
    present, n_cells, sums = pseudobulk(rows, cols, dense[rows, cols], codes, 7)
    assert present.tolist() == [1, 4, 9] and n_cells.tolist() == [4, 5, 3] and sums.dtype == np.uint32
    for i, label in enumerate(present):
        np.testing.assert_array_equal(sums[i], dense[codes == label].sum(axis=0))


def test_pseudobulk_can_restrict_to_included_cells():
    dense = np.arange(24).reshape(6, 4) % 5
    codes = np.array([0, 0, 1, 1, 2, 2])
    include = np.array([True, False, True, True, False, False])  # e.g. left-hemisphere cells
    rows, cols = np.nonzero(dense)
    present, n_cells, sums = pseudobulk(rows, cols, dense[rows, cols], codes, 4, include)
    assert present.tolist() == [0, 1] and n_cells.tolist() == [1, 2]
    np.testing.assert_array_equal(sums, [dense[0], dense[2] + dense[3]])
