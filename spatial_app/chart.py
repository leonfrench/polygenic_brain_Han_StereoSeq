"""Compact, acknowledged score updates for the multi-slice cell viewer."""
from pathlib import Path
import base64
import time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from spatial_app.layout import slice_grid, dot_size, quantize

_component = components.declare_component('spatial_cells', path=str(Path(__file__).with_name('cell_component')))


def encode_array(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode('ascii')


def label_codes(values):
    """Sorted labels present and each row's code, as np.unique; categories skip sorting a million strings."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.cat.remove_unused_categories()
        values = values.cat.reorder_categories(sorted(values.cat.categories))
        return values.cat.categories.astype(str).to_numpy(), values.cat.codes.to_numpy()
    return np.unique(values.astype(str).to_numpy(), return_inverse=True)


def geometry_payload(cells, slices, annotations, columns=None, um_per_unit=0.5, noun='slices', items='cells', marker='dot', dot_scale=1.0):
    """Positions, slice panels and every annotation level for cells in drawing order.

    um_per_unit converts coordinate units to micrometres for the dot-size note
    (0.5 for Stereo-seq spots); noun names the panels in the viewer's status line and
    items the points. marker='square' draws each point as a square one dot wide, for
    bins on a regular grid (whose nearest-neighbour spacing, the dot, is the bin side).
    Squares keep float32 positions instead of 16-bit ones, so neighbours stay edge to edge
    at any zoom; bins are few, so the payload stays small. dot_scale enlarges (or shrinks) the dots
    relative to that spacing; the status line reports it.
    """
    px, py, panels, width, height = slice_grid(cells, slices, columns=columns)
    positions = '<f4' if marker == 'square' else '<u2'
    qx, qy, step = (px, py, 1.0) if marker == 'square' else quantize(px, py, width, height)
    levels = {}
    for column, level in annotations.items():
        labels, codes = label_codes(cells[column])
        dtype = '<u1' if len(labels) <= 256 else '<u2'
        levels[column] = dict(title=level['title'], kind=level['kind'], labels=labels.tolist(),
                              colors=[level['labels'].get(label, '#B0B0B0') for label in labels],
                              descriptions=[level['descriptions'].get(label, '') for label in labels],
                              codes=encode_array(codes, dtype), dtype=dtype)
    order = {s: i for i, s in enumerate(slices)}
    return dict(n=len(cells), width=width, height=height, step=step, dot=dot_size(cells)*dot_scale, dot_scale=float(dot_scale),
                um_per_unit=float(um_per_unit), noun=noun, items=items, marker=marker, panels=panels, layout='|'.join(slices),
                px=encode_array(qx, positions), py=encode_array(qy, positions), positions=positions,
                slice=encode_array(cells.slice.map(order).to_numpy(), '<u1'), levels=levels)


def score_patch(previous, scores):
    if previous is None:
        indices = np.flatnonzero(np.isfinite(scores))
    else:
        indices = np.flatnonzero(~((previous == scores) | (np.isnan(previous) & np.isnan(scores))))
    return dict(indices=encode_array(indices, '<u4'), values=encode_array(scores[indices], '<f8'))


@st.cache_resource(max_entries=2, show_spinner=False)
def shared_geometry(version, _cells, slices, _annotations, um_per_unit, noun, items='cells', marker='dot', dot_scale=1.0):
    """One geometry payload per data version, shared by every session instead of rebuilt and held by each."""
    return geometry_payload(_cells, list(slices), _annotations, um_per_unit=um_per_unit, noun=noun, items=items, marker=marker, dot_scale=dot_scale)


def cell_chart(cells, scores, *, slices, annotations, version, mode, cell_level, region_level,
               cell_aurocs, region_aurocs, filename, highlight=None, clip=0.01, um_per_unit=0.5, noun='slices',
               hover=None, absolute=False, items='cells', marker='dot', dot_scale=1.0, pixelated=False, key='cell_plot'):
    """Cells must already be in drawing order; scores align with those rows.

    highlight is an optional {label, seq} request; the viewer applies each new seq once,
    so legend clicks in the browser are not overridden on later reruns. Colour scales
    span the clip and 1 - clip quantiles: per slice for gene-set AUROC (arbitrary units),
    or over all points shown with AUROC values on the legend when absolute is set (for a
    single slice, where no batch effects separate slices, or for bins on one scale across
    slices). hover is an optional {levels, slice_with, brackets}
    tooltip spec: those annotation levels plus the chosen cell type and region levels, each
    label with a description shown as 'description (label)', the slice on the slice_with
    line, and brackets ({level: other level}, optional) putting another level's label in a
    level's brackets instead of on its own line. Without it the tooltip shows the slice,
    the two chosen levels and the slice's colour range. items and marker are as for geometry_payload.
    pixelated has the browser scale the plot canvas by nearest neighbour (CSS image-rendering: pixelated),
    so square edges stay hard where the canvas is resampled (display scaling or browser zoom).
    """
    started = time.perf_counter()
    state_key = key + '_transport'
    ack = st.session_state.get(key) or {}
    state = st.session_state.get(state_key)
    if state is None or state['version'] != version or 'history' not in state:
        state = dict(version=version, geometry=shared_geometry(version, cells, tuple(slices), annotations, um_per_unit, noun, items, marker, dot_scale), seq=0,
                     pending=None, baseline=None, client=None, history={})
        st.session_state[state_key] = state
    client = ack.get('client')
    if client != state['client']:
        state.update(client=client, baseline=None, pending=None, history={})
    pending = state['pending']
    acknowledged = state['history'].get(ack.get('seq')) if ack.get('version') == version else None
    if acknowledged is not None:
        state['baseline'] = acknowledged
    # Keep one pending snapshot; resend it until acknowledged. A missed update
    # is recovered by an absolute patch from the last acknowledged baseline.
    if pending is None or not np.array_equal(pending['scores'], scores, equal_nan=True):
        state['seq'] += 1
        packet = score_patch(state['baseline'], scores)
        pending = dict(seq=state['seq'], scores=scores.copy(), packet=packet,
                       reset=state['baseline'] is None)
        state['pending'] = pending
        state['history'][pending['seq']] = pending['scores']
        while len(state['history']) > 4:
            del state['history'][next(iter(state['history']))]
    geometry = None if ack.get('version') == version and client else state['geometry']
    packet = dict(version=version, seq=pending['seq'], reset=pending['reset'], **pending['packet'])
    finite = lambda values: {name: float(v) for name, v in values.items() if np.isfinite(v)}
    payload_bytes = sum(len(packet[k]) for k in ('indices', 'values'))
    if geometry:
        payload_bytes += len(geometry['px'])*2 + len(geometry['slice']) + sum(len(l['codes']) for l in geometry['levels'].values())
    _component(geometry=geometry, packet=packet, mode=mode, cell_level=cell_level, region_level=region_level,
               cell_aurocs=finite(cell_aurocs), region_aurocs=finite(region_aurocs),
               highlight=highlight, clip=clip, filename=filename, hover=hover, absolute=bool(absolute), pixelated=bool(pixelated),
               key=key, default=None)
    print(f"[timing] chart={filename} compact_update_s={time.perf_counter()-started:.3f} "
          f"payload_mib={payload_bytes/1024**2:.3f} geometry_sent={geometry is not None}", flush=True)
