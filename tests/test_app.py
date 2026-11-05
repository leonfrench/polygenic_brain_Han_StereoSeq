from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from spatial_app.enrichment import MIN_CELLS
from spatial_app.bins import MIN_BIN_CELLS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/mouse1'
BINS = ROOT/'data/mouse1-bins178um'
pytestmark = pytest.mark.skipif(not (DATA/'manifest.json').exists(), reason='prepared dataset not available')


@pytest.fixture
def manifest():
    return json.loads((DATA/'manifest.json').read_text())


def finish(app, manifest):
    """Advance the fragment timer until every batch of every slice is scored."""
    batches = {b for s in manifest['slices'] for b in s['batches']}
    for _ in range(len(batches)+5):  # one batch per rerun
        if batches <= app.session_state['done']:
            return
        app.run()
        assert not app.exception
    raise AssertionError('scoring did not finish')


def viewer_args(app):
    """The arguments the app last sent to the cell viewer."""
    return json.loads(app.get('component_instance')[0].proto.json_args)


def test_all_slices_score_progressively_and_fill_both_tables(manifest):
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run()
    assert not app.exception
    scores = app.session_state['scores']
    assert len(scores) == manifest['n_cells']
    # Batches interleave across slices, so the first batch comes from the first slice.
    first = manifest['slices'][0]
    assert app.session_state['done'] == {first['batches'][0]}
    assert np.isfinite(scores).sum() == next(f['n_cells'] for f in manifest['files'] if f['path'] == first['batches'][0])
    assert app.session_state['job']['query'].universe_size == manifest['n_genes']
    # A new query first clears the tables; the next update fills them, long before scoring ends.
    assert not app.dataframe and any('Calculating expression specificity' in i.value for i in app.info)
    app.run()
    early = [app.dataframe[0].value, app.dataframe[1].value]
    finish(app, manifest)
    assert np.isfinite(app.session_state['scores']).all()
    cell_table, region_table = app.dataframe[0].value, app.dataframe[1].value
    pd.testing.assert_frame_equal(cell_table, early[0]); pd.testing.assert_frame_equal(region_table, early[1])
    cells = pd.read_parquet(DATA/'cells.parquet')
    left = cells[cells.hemisphere.eq('L')]  # region profiles use the finely annotated left hemisphere
    for table, column, source in [(cell_table, 'cell_subclass', cells), (region_table, 'region_area', left)]:
        counts = source[column].value_counts()
        assert (table['Cells'] >= MIN_CELLS).all()
        assert table.set_index(table.columns[0])['Cells'].to_dict() == counts[counts >= MIN_CELLS].to_dict()
    assert cell_table.columns[0] == 'Cell subclass' and region_table.columns[:4].tolist() == ['Acronym', 'Full name', 'Cells', '# slices']
    slices = left.groupby('region_area', observed=True).slice.nunique()  # slices contributing left-hemisphere cells
    assert region_table.set_index('Acronym')['# slices'].to_dict() == slices.reindex(region_table['Acronym']).to_dict()
    assert '# slices' not in cell_table
    assert not {'Isocortex', 'CB', 'MB', 'OLF', 'P'} & set(region_table['Acronym'])  # right-only whole-division labels
    # The default set (Tac2, Gng8, Chat, Slc18a3, Chrnb4, ...) marks the medial habenula.
    assert app.session_state['job']['query'].target_symbols == ('CHAT', 'CHRNB4', 'GNG8', 'GPR151', 'HTR2C', 'POU4F1', 'SLC18A3', 'TAC1', 'TAC2')
    assert 'MH' in region_table.iloc[:5, 0].tolist()


def test_all_slices_are_always_shown_and_levels_change_tables(manifest):
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run()
    assert not app.multiselect  # no slice selector
    header = next(m.value for m in app.markdown if 'coronal slices' in m.value)
    assert header == f"**Mouse 1 · {len(manifest['slices'])} coronal slices**"  # the panels carry the slice IDs
    panels = app.session_state['cell_plot_transport']['geometry']['panels']
    assert len({p['y'] for p in panels}) == 3 and len({p['x'] for p in panels}) == 6  # three rows of six (the last has four)
    assert viewer_args(app)['max_height'] == 440  # no taller than the viewer's minimum
    finish(app, manifest)
    app.selectbox(key='cell_level').set_value('cell_cluster').run()
    app.selectbox(key='region_level').set_value('region').run()
    assert not app.exception
    assert app.dataframe[0].value.columns[:2].tolist() == ['Cell cluster', 'Cell subclass']
    assert app.dataframe[1].value.columns[0] == 'Acronym' and app.subheader[1].value == 'Region specificity · finest region'
    for mode in ['Cell type', 'Region', 'Cell type AUROC', 'Region AUROC']:
        app.radio(key='colour_mode').set_value(mode).run()
        assert not app.exception


def test_new_gene_set_switches_colours_back_to_gene_set_auroc():
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run()
    app.radio(key='colour_mode').set_value('Region').run()
    app.selectbox(key='region_level').set_value('region').run()
    assert app.radio(key='colour_mode').value == 'Region'  # other controls keep the mode
    app.text_area(key='target_input').set_value('Gfap Aqp4 Aldh1l1').run()
    assert not app.exception
    assert app.radio(key='colour_mode').value == 'Gene-set AUROC'


@pytest.mark.skipif(not (BINS/'manifest.json').exists(), reason='binned dataset not available')
def test_bins_are_one_more_colour_mode_and_rescored_for_a_new_gene_set():
    bins = json.loads((BINS/'manifest.json').read_text())
    mode = f"{bins['bin_um']:g}µm bin AUROC"
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run()
    assert app.radio(key='colour_mode').options == ['Gene-set AUROC', 'Cell type AUROC', 'Region AUROC', mode, 'Cell type', 'Region']
    app.radio(key='colour_mode').set_value(mode).run()
    assert not app.exception
    ((key, (values, reference)),) = app.session_state['bin_scores'].items()
    assert key[1] == bins['created_utc'] and reference is None  # keyed by the bins build too, and scored without error
    n_cells = pd.read_parquet(BINS/'bins.parquet').sort_values('draw_order').n_cells.to_numpy()
    assert len(values) == bins['n_bins'] and np.array_equal(np.isfinite(values), n_cells >= MIN_BIN_CELLS)
    assert ((values[np.isfinite(values)] >= 0) & (values[np.isfinite(values)] <= 1)).all()
    assert any(c.value.startswith(f"Each square is a {bins['bin_um']:g} µm bin") for c in app.caption)
    geometry = app.session_state['cell_plot_transport']['geometry']
    assert len({p['y'] for p in geometry['panels']}) == 3 and viewer_args(app)['max_height'] == 440  # as for the cells
    app.text_area(key='target_input').set_value('Gfap Aqp4 Aldh1l1').run()
    assert not app.exception and app.radio(key='colour_mode').value == 'Gene-set AUROC'  # a new gene set starts on the cells
    assert any(c.value.startswith('Each dot is one segmented cell') for c in app.caption)
    app.radio(key='colour_mode').set_value(mode).run()
    ((_, (astrocyte, _)),) = app.session_state['bin_scores'].items()  # scores for the current query only
    assert not app.exception and not np.allclose(astrocyte, values, equal_nan=True)


@pytest.mark.skipif(not (BINS/'manifest.json').exists(), reason='binned dataset not available')
def test_a_failed_bin_scoring_shows_only_a_reference_and_is_not_retried(tmp_path, monkeypatch):
    bins = json.loads((BINS/'manifest.json').read_text())
    (tmp_path/'ranks').mkdir()
    for name in ['manifest.json', 'bins.parquet', *(s['ranks'] for s in bins['slices'][:-1])]:  # the last slice's ranks are missing
        (tmp_path/name).symlink_to(BINS/name)
    monkeypatch.setenv('MOUSE_APP_BINS', str(tmp_path))
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run()
    app.radio(key='colour_mode').set_value(app.radio(key='colour_mode').options[3]).run()  # the bin mode
    assert not app.exception
    ((_, (values, reference)),) = app.session_state['bin_scores'].items()
    assert reference and np.isnan(values).all()
    assert [e.value for e in app.error] == [f'Unable to score the bins (reference {reference}).']  # no paths or exception text
    app.run()  # the next fragment tick shows the same failure instead of scoring again
    assert [r for _, (_, r) in app.session_state['bin_scores'].items()] == [reference]
    app.radio(key='colour_mode').set_value('Gene-set AUROC').run()
    assert not app.exception and any(c.value.startswith('Each dot is one segmented cell') for c in app.caption)


def test_new_gene_set_clears_the_tables_before_showing_its_own():
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run().run()
    before = app.dataframe[1].value
    app.text_area(key='target_input').set_value('Pvalb Sst Vip Gad1 Gad2').run()
    assert not app.dataframe  # the previous set's tables are gone at once
    app.run()
    after = app.dataframe[1].value
    assert after.iloc[0, 0] != before.iloc[0, 0] and not after.equals(before)


def test_gene_matching_lists_used_targets_and_excluded_symbols():
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run()
    app.text_area(key='target_input').set_value('Tac2 Chat NOT_A_GENE').run()
    app.text_area(key='background_input').set_value('Tac2 Gfap Aqp4 ALSO_MISSING').run()
    assert not app.exception
    matching = next(e for e in app.expander if e.label.startswith('Gene matching'))  # one panel for used and excluded genes
    assert matching.label == 'Gene matching · 1 target used in a 3-gene universe · 2 target symbols excluded'
    assert matching.markdown[0].value == '**Targets used:** Tac2'
    assert [w.value for w in matching.warning] == ['Targets not in dataset: NOT_A_GENE', 'Background genes not in dataset: ALSO_MISSING',
                                                   'Targets outside background: CHAT']
    assert not any(e.label.startswith('Genes used') for e in app.expander)


def test_human_symbols_background_and_invalid_queries():
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run()
    app.selectbox(key='input_species').set_value('Human').run()
    app.text_area(key='target_input').set_value('MBP PLP1 NOT_A_GENE').run()
    assert not app.exception
    assert app.session_state['mapping_unmapped'] == (['NOT_A_GENE'], [])
    assert app.session_state['job']['query'].target_symbols == ('MBP', 'PLP1')
    app.text_area(key='background_input').set_value('MBP PLP1 SNAP25 GAPDH').run()
    assert app.session_state['job']['query'].universe_size == 4
    matching = next(e for e in app.expander if e.label.startswith('Gene matching'))  # mouse orthologs inside the background
    assert matching.label == 'Gene matching · 2 targets used in a 4-gene universe'
    assert matching.markdown[0].value == '**Targets used:** Mbp, Plp1'
    # GAPDH has two mouse orthologs; Gm20899 is not in the dataset.
    assert [w.value for w in matching.warning] == ['Background genes not in dataset: GM20899']
    app.text_area(key='background_input').set_value('MBP PLP1').run()
    assert app.session_state['job'] is None and 'non-target' in app.session_state['query_error']
    app.text_area(key='target_input').set_value('NOT_A_GENE').run()
    assert app.session_state['job'] is None and app.error
