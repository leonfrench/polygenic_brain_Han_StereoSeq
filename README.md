# Polygenic gene-set tester for the Han et al. Stereo-seq mouse brain

A Streamlit app for the Stereo-seq whole mouse brain atlas of Han et al. (2025; mouse 1, whole transcriptome). Enter a
gene set; every segmented cell gets an exact set-versus-complement AUROC from its gene expression ranks, shown on 16
coronal slices in two rows of eight, anterior to posterior. The app also tests whether the gene set is specifically
expressed in each **cell type** and each **brain region**, using pseudobulk profiles as in the
[SEA-AD polygenic tester](https://github.com/leonfrench/SEA_AD_multi_polygenic_app). It is a cell-based relative of the
[spatial fetal app](https://github.com/leonfrench/spatial_fetal_app). A
[MERFISH version](https://github.com/leonfrench/polygenic_brain_Yao_MERFISH500) does the same for the Allen Brain Cell
Atlas whole mouse brain (500-gene panel).

## Run locally

```bash
~/miniconda3/bin/conda create -y -p ./.conda-env -c conda-forge --override-channels python=3.12 pip
./.conda-env/bin/python -m pip install -r requirements-prep.txt pytest
./.conda-env/bin/streamlit run app.py
```

Open http://localhost:8501. `requirements.txt` is enough to run the app on the prepared data in `data/mouse1/`;
`requirements-prep.txt` adds polars for preprocessing. `MOUSE_APP_DATA=/path/to/export` selects a different prepared
dataset.

## Deploy on Streamlit Community Cloud

- Main file: `app.py`. Python 3.12 is the development version; the app and its tests also pass on Python 3.14 with a
  clean install of `requirements.txt` (the preprocessing tests are skipped there). `requirements.txt` is the dependency
  file Community Cloud installs; do not add an `environment.yml`, which it would install instead.
- The prepared data (1.3 GB in 89 files; every file is under GitHub's 100 MB limit, the largest is 49 MB) is committed in
  plain git, so no Git LFS is needed. The first deploy clones all of it.
- Memory: Community Cloud allows up to 2.7 GB. A local server held about 0.4 GB at rest and peaked at 1.4 GB while a
  browser session scored every cell and switched through all levels and colour modes; a second concurrent session did
  not raise the peak. Only the current score batch is held in memory, label columns load as categories, and the plot
  geometry is built once and shared by all sessions.
- The first page load sends about 25 MB of plot data to the browser (13 MB of geometry, then 12 MB of scores as batches
  finish).

## Using the app

- **Slices**: every processed slice is shown, anterior to posterior (rows of four for up to 12 slices, rows of
  eight beyond). Batches from all slices are scored in interleaved order, so all panels fill in together. To
  change slices, reprocess with a different set of files.
- **Genes**: mouse symbols by default (matched case-insensitively). Human or rhesus macaque symbols are
  converted to mouse orthologs with the bundled HomoloGene snapshot (`data/homologene_to_human.csv.gz`);
  one-to-many orthologs are all kept and unmapped symbols are reported. An optional background is the
  complete universe, including the targets; ranks are recomputed inside it. **Genes used** lists the matched
  targets, and its download gives every gene of the scoring universe with its role (target or background).
- **Colour cells by**: gene-set AUROC, cell type, region, cell type AUROC or region AUROC (the last two
  colour each cell by its label's specificity AUROC from the tables). The cell type level
  (class / subclass / cluster) and region level (division / area / finest) set both the colours and the tables.
- **Plot**: one panel per slice, all sharing the same registered frame and scale, so slices keep their
  registered position and relative size, on a black background. Dots are sized to the median nearest-cell
  spacing (~14 µm). AUROC ranges differ between slices (batch effects), so gene-set AUROC colours are relative
  within each slice: each slice is scaled between its own 1st and 99th percentiles, which are shown above its panel,
  and the legend is in arbitrary units. Hover gives the exact AUROC. The label AUROC modes (one pooled value per
  label) share one scale clipped at the 1st and 99th percentiles of the cells shown. The clip is `COLOUR_CLIP` in `app.py`. Colours only; the statistics
  are unchanged. Drag to
  pan, scroll to zoom, double-click to reset, click a legend entry to highlight that label, and save the current
  view as PNG. Hover shows a cell's gene-set AUROC, then its subclass (cluster in brackets), its brain area (name,
  then acronym in brackets) with the slice, and its finest region; the class or division is added when that level is
  chosen in the sidebar.
- **Tables**: cell type and region expression specificity across all slices, with CSV downloads, plus
  a download of every cell's score once scoring completes. The tables use pseudobulk profiles, not
  cell scores, so they appear as soon as the gene set is valid. Tick a row's checkbox to colour the plot by that
  table's specificity AUROC (cell type or region) and highlight the label; untick it to return to the previous
  colour mode.

## Data

Raw inputs live in `data/Han et al./` (not version controlled):

| File | Contents |
| --- | --- |
| `total_gene_<SLICE>_mouse_f001_2D_mouse1-20230119.txt.gz` | One row per spot × gene: `gene, x, y, umi_count, cell_label, gene_area, rx, ry` |
| `stereoseq.celltypeTransfer.2mice.all.tsv.gz` | Transferred cell type per `mouse, section_id, cell_id`: 308 clusters, 19 subclasses, 6 classes, cluster colours |
| `regions-mouse1-20230119.rmDup.tsv` | `gene_area` code → hemisphere-prefixed CCF acronym path (`LOLF-MOB-MOBgl`) and description |

What the inspection showed:

- `cell_label` 0 means the spot is not in a segmented cell. A cell spans tens of spots, and the same gene
  can appear on several of them, so counts are summed per cell × gene.
- `rx, ry` are an exact rigid transform (rotation with reflection, RMSE 0.29) of the chip `x, y`. They are the
  registered coordinates, and with y pointing down they show dorsal at the top. They are used for display.
- Positive region codes are the left hemisphere (465 fine-grained labels, with cortical layers), negative codes are
  the right hemisphere (61 coarser labels), and 0 is background. Some author notes in the region table are in
  Chinese (for example L6a/L6b not distinguished); they are translated in the region descriptions.
- About 2–6% of segmented cells per slice have no transferred cell type; they have low UMI counts
  (median 637 versus a typed-cell minimum of 134) and are excluded.

## Preprocessing

```bash
./.conda-env/bin/python scripts/prepare_mouse_cells.py --force
```

With no `--slices`, every `total_gene_*` file in `data/Han et al./` is processed. To change slices, put the
new files in that folder and rerun with the full list: all processed slices share one gene universe, so
their AUROCs and the pooled enrichment tests are comparable. The current 16 slices (15 GB of gzip input) take
15–18 minutes, with peak memory of 16–20 GB. Each file is decompressed to a temporary text file and aggregated with
the polars streaming engine, so the largest slice (T305, ~7.5 GB of text, 219M rows) never sits in memory as a whole.
The first run downloads the MGI marker lists and the Allen structure graph into `data/reference/`. `--force`
replaces an existing output directory once the new one is complete.

Steps:

1. **Cells**: spot UMIs summed per cell × gene; centroid of the cell's unique spots in `rx, ry` (and `x, y`);
   region = the `gene_area` code with the largest UMI share (ties go to the lower code; the share is kept as
   `region_fraction`, median 1.0). Only cells with a transferred cell type are kept.
2. **Genes**: current MGI protein-coding symbols, then unambiguous withdrawn symbols (the data use an older
   annotation; 2,021 of the retained source symbols map this way). No synonym guessing. Symbols that map to the same
   gene are summed before ranking. The universe is the protein-coding genes detected in at least
   `--min-cells` (default 0) typed cells of the processed slices.
3. **Ranks**: average ranks within each cell, with all undetected genes tied at the bottom, stored as
   doubled `uint16` values (format `sparse-doubled-average-ranks-v1`, unchanged from the fetal app).
   Thirty-two random cells per slice are checked against `scipy.stats.rankdata` and against a pairwise
   AUROC, and every batch is checked after a write/read round trip.
4. **Batches**: each slice's cells are shuffled (seed 20261016) and split into equal batches of at most 16,384
   cells (`--batch-cells`) and 12 million stored cell × gene entries (`--batch-entries`), in
   `progressive/<SLICE>/NNNNN.npz`. The entry limit keeps the denser posterior slices under 50 MB per file.
5. **Annotations**: `annotations.json` holds labels and colours for each level. Cluster colours come from the
   transfer table. Subclass and class take the colour of their most numerous cluster. Regions use Allen CCF
   colours (inheriting the parent's colour for non-Allen acronyms such as `CAsr`). Labels with near-identical
   colours, such as cortical layers, are spread in lightness so they stay distinguishable.

Output (`data/mouse1/`, 1.41 GB for 16 slices, of which 81 MB is pseudobulk; largest file 49 MB):

| File | Contents |
| --- | --- |
| `manifest.json` | Universe size, per-slice cell ranges, batch paths, input and reference checksums, mapping counts, validation |
| `cells.parquet` | `cell_index`, `slice`, `cell_id`, `source_obs_id`, `rx, ry, x, y`, UMI and gene counts, cell type levels and cluster colour, `region_code`, `hemisphere`, region levels, `region_fraction`, `draw_order`, `batch` |
| `genes.parquet` | `gene_index`, MGI symbol, ID and name, detection counts, merged source symbols |
| `gene_detection.parquet` | Detected-cell counts per gene and slice (for the sparsity note) |
| `pseudobulk/<SLICE>.npz` | For each annotation level: summed retained-gene UMIs per (label, gene) for the labels present (`<level>_counts`, `uint32`), their global label indices (`<level>_index`, into `manifest.json` `pseudobulk.labels`) and cell counts (`<level>_cells`). Region levels include left-hemisphere cells only (`pseudobulk.cells` in the manifest) |
| `gene_mapping.parquet` | Every source symbol with its MGI mapping and status |
| `annotations.json` | Titles, labels, colours and region descriptions for all six annotation levels |

| Slice | Segmented cells | Typed cells kept | Batches |
| --- | ---: | ---: | ---: |
| T241 | 6,676 | 6,450 | 1 |
| T250 | 19,750 | 19,154 | 2 |
| T258 | 29,060 | 28,347 | 2 |
| T266 | 26,138 | 25,181 | 3 |
| T273 | 33,205 | 31,869 | 4 |
| T281 | 38,115 | 36,992 | 4 |
| T285 | 51,030 | 49,625 | 5 |
| T294 | 62,115 | 60,756 | 4 |
| T305 | 68,770 | 66,780 | 6 |
| T309 | 71,222 | 69,280 | 7 |
| T319 | 72,019 | 70,573 | 6 |
| T330 | 65,627 | 63,486 | 4 |
| T340 | 58,806 | 56,988 | 5 |
| T346 | 47,449 | 44,417 | 4 |
| T355 | 61,651 | 60,874 | 5 |
| T368 | 51,441 | 50,499 | 5 |
| **Total** | **763,074** | **741,271** | **67** |

20,193 protein-coding genes are retained (from 27,820 source symbols: 7,148 not protein-coding,
455 unmapped, 2 ambiguous withdrawn symbols). All slices share the registered frame: from 3.2 × 1.6 mm (T241,
olfactory bulb tip) to 7.1 × 6.5 mm (T319), at 0.5 µm per unit.

## Cell type × region cross-tabulation

```bash
./.conda-env/bin/python scripts/crosstab_annotations.py
```

Writes `analysis/celltype_region_counts.csv` (long counts over slice, hemisphere, all region levels and all
cell type levels), `subclass_by_division.csv` and `subclass_by_area.csv` (subclass × slice × region), and
`marker_detection.csv` (percentage of cells detecting OB and cortical markers, by subclass, slice and
OLF versus elsewhere). For the current 16 slices, the transferred labels are not fully consistent with the
regions: 1,786 "Olfactory bulb excitatory" and 6,209 "Olfactory bulb inhibitory" cells lie outside the
olfactory areas, the excitatory ones with essentially no mitral/tufted markers (`Cdhr1` 3.0% and `Tbx21` 0.4%,
versus 56% and 9% inside), while 22,193 of the 57,504 main olfactory bulb cells carry cortical "Telencephalon
excitatory" labels. The authors' whole-brain transfer table shows the same
0.2–0.5% OB excitatory labels in sections T277–T345, so these come from the source annotation, not preprocessing.

## Methods

**Cell scores (plot).** For k target genes in a universe of G genes, a cell's score is
`AUROC = (sum of target ranks − k(k+1)/2) / (k(G − k))`, the probability that a target gene outranks a
non-target gene in that cell, with half credit for ties.

**Expression specificity (tables).** For the chosen level, each label's pseudobulk sums the UMIs of its cells
across all slices and is converted to log₂(CPM + 1). Region pseudobulks use left-hemisphere cells only: the right
hemisphere is annotated mostly as whole structures (its most common finest label is `Isocortex`), which would add
division-sized profiles to the area table and mix coarse and fine cells under shared labels. Cell type pseudobulks
use all cells. Each gene is z-scored across the labels, removing its
average expression (genes that do not vary get z = 0). Within each label, genes are ranked by z-score (average
ranks for ties), and the AUROC compares the target genes with the rest of the universe: above 0.5 means the gene
set is more specific to that label than other genes. With a custom background, genes are ranked within that
universe. Labels with fewer than 20 cells are not profiled. Two-sided Mann–Whitney p-values
over genes (normal approximation with tie and continuity correction, matching `scipy.stats.mannwhitneyu`) are
adjusted with Benjamini–Hochberg within each table (pFDR). Genes are not independent (co-expression), so the p-values
are exploratory. This follows the [SEA-AD polygenic tester](https://github.com/leonfrench/SEA_AD_multi_polygenic_app),
which ranks genes by pseudobulk z-scores in the same way; its p-values omit the tie correction.

## Tests

```bash
./.conda-env/bin/python -m pytest tests -q
node tests/test_cells.cjs
```

These cover rank encoding and scoring against SciPy and pairwise definitions, pseudobulk sums,
specificity z-scores and AUROCs against `mannwhitneyu`, ortholog mapping, region parsing and translation, the slice grid
layout, the score transport protocol, the canvas helpers, and end-to-end app runs on `data/mouse1`. With only
`requirements.txt` installed, the preprocessing tests are skipped.

## Data credit

Han et al. *Single-cell spatial transcriptomic atlas of the whole mouse brain.*
Neuron. [doi:10.1016/j.neuron.2025.02.015](https://doi.org/10.1016/j.neuron.2025.02.015).
Derived from Han et al. ([Brain Science Data Center](https://www.braindatacenter.cn/datacenter/web/#/dataSet/details?id=1888062851995131906)),
licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Spot counts were summed per cell, and
cells and genes were filtered for this app.
Gene symbols: [MGI](https://www.informatics.jax.org/). Region colours: Allen Mouse Brain Atlas structure graph.

