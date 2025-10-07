from typing import List, Dict
from .utils.geometry import overlap_area

# Represent a cell as dict: { 'bbox':[x1,y1,x2,y2], 'texts':[], 'value':None, 'conf':0.0 }

def assign_cells(words: List[Dict], grid_rows: List[List[list]]):
    # Build cell objects
    cells = []
    for row in grid_rows:
        cell_row = [{'bbox': cbox, 'texts': [], 'value': None, 'conf': 0.0} for cbox in row]
        cells.append(cell_row)

    # Assign words to the single best-overlap cell
    flat_cells = [(ri, ci, c) for ri, row in enumerate(cells) for ci, c in enumerate(row)]
    for w in words:
        best = None
        best_area = 0.0
        for _, _, c in flat_cells:
            area = overlap_area(w['bbox'], c['bbox'])
            if area > best_area:
                best_area = area
                best = c
        if best is not None and best_area > 0:
            best['texts'].append(w)

    # Order texts in each cell
    for row in cells:
        for c in row:
            c['texts'].sort(key=lambda w: (w['bbox'][1], w['bbox'][0]))
    return cells
