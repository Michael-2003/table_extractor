from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from .config import Config


cfg = Config()

# -------- Helpers --------


def _cy(box):
    x1,y1,x2,y2 = box
    return 0.5*(y1+y2)


def _cx(box):
    x1,y1,x2,y2 = box
    return 0.5*(x1+x2)

# -------- Row bands from Y-clustering --------


def build_row_bands(words: List[Dict]) -> List[Tuple[int,int]]:
    if not words:
        return []
    ys = np.array([[ _cy(w['bbox']) ] for w in words])
    clustering = DBSCAN(eps=cfg.row_cluster_eps, min_samples=1).fit(ys)
    bands = []
    for lbl in sorted(set(clustering.labels_)):
        rows = [w for w,lab in zip(words, clustering.labels_) if lab==lbl]
        y1 = int(min(w['bbox'][1] for w in rows))
        y2 = int(max(w['bbox'][3] for w in rows))
        pad = max(2, int(0.2*(y2-y1+1)))
        bands.append((max(0,y1-pad), y2+pad))
    bands.sort(key=lambda b: b[0])
    return bands    


# -------- Column candidates from x-edge histograms --------
def propose_column_breaks(words: List[Dict]) -> List[int]:
    xs = []
    for w in words:
        x1,_,x2,_ = w['bbox']
        xs.append(x1); xs.append(x2)
    xs = np.array(xs)
    xs.sort()
    # find valleys as large gaps
    gaps = xs[1:] - xs[:-1]
    # choose gaps greater than threshold
    idx = np.where(gaps >= cfg.min_col_gap_px)[0]
    cuts = [int((xs[i]+xs[i+1])//2) for i in idx]
    # deduplicate nearby cuts
    uniq = []
    for c in cuts:
        if not uniq or abs(c-uniq[-1])>cfg.min_col_gap_px:
            uniq.append(c)
    return uniq

# -------- Per-row cuts (greedy PoC) --------


def row_cuts_for_band(words: List[Dict], band: Tuple[int,int], cuts: List[int], rtl=False) -> List[int]:
    y1,y2 = band
    in_band = [w for w in words if not (w['bbox'][3] < y1 or w['bbox'][1] > y2)]
    in_band.sort(key=lambda w: _cx(w['bbox']), reverse=rtl)
    # Greedy: keep cuts that fall between sufficiently separated consecutive words
    xs = [(_cx(w['bbox']), w) for w in in_band]
    xs.sort()
    chosen = []
    for i in range(len(xs)-1):
        right_edge = xs[i][1]['bbox'][2]
        left_edge = xs[i+1][1]['bbox'][0]
        gap = left_edge - right_edge
        mid = (right_edge + left_edge)//2
        if gap >= cfg.min_col_gap_px:
    # if a proposed cut is near mid, accept it
            nearest = min(cuts, key=lambda c: abs(c-mid)) if cuts else mid
            if abs(nearest - mid) <= cfg.min_col_gap_px:
                chosen.append(nearest)
            else:
                chosen.append(mid)
    # dedup
    chosen2 = []
    for c in chosen:
        if not chosen2 or abs(c-chosen2[-1])>cfg.min_col_gap_px:
            chosen2.append(c)
    return chosen2


# -------- Build grid polygons --------


def build_grid(words: List[Dict], rtl=False):
    bands = build_row_bands(words)
    cuts_global = propose_column_breaks(words)
    grid_rows = []
    for b in bands:
        row_c = row_cuts_for_band(words, b, cuts_global, rtl=rtl)
        # Convert cuts to cell x-intervals
        xs = [-10**9] + row_c + [10**9]
        cells = []
        for i in range(len(xs)-1):
            x1 = xs[i]
            x2 = xs[i+1]
            cells.append([x1, b[0], x2, b[1]])
        grid_rows.append(cells)
    return grid_rows


# API

def segment_structure(words: List[Dict], rtl=False):
    """Return grid as list of rows, each row is list of cell bboxes [x1,y1,x2,y2]."""
    return build_grid(words, rtl=rtl)