import csv,hashlib,struct
from pathlib import Path
IMAGE_WIDTH=IMAGE_HEIGHT=4096

def mrc_shape(path):
    with path.open('rb') as f:
        header = f.read(16)
    if len(header) < 16:
        raise ValueError(f'truncated MRC header: {path}')
    nx, ny, nz, mode = struct.unpack('<4i', header)
    if not (0 < nx < 100000 and 0 < ny < 100000 and (0 < nz < 100000) and (mode in (0, 1, 2, 3, 4, 6, 12))):
        raise ValueError(f'invalid MRC header: {path}: {(nx, ny, nz, mode)}')
    return (nx, ny, nz, mode)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def cryolo_log_confirms_zero_candidates(log_path):
    """crYOLO 1.9.9 writes a zero-byte CBOX for zero boxes; require its success summary."""
    import re
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding='utf-8', errors='replace')
    return re.search('\\b0 particles in total has been found\\b', text) is not None

def parse_cryolo_cbox(text, source='<CBOX>'):
    """Parse STAR loops, selecting coordinate loops and ignoring metadata loops."""
    import shlex
    lines = text.splitlines()
    required = ('_CoordinateX', '_CoordinateY', '_Width', '_Height', '_Confidence')
    coordinate_rows = []
    coordinate_loops = 0
    i = 0
    while i < len(lines):
        if lines[i].strip() != 'loop_':
            i += 1
            continue
        i += 1
        headers = []
        while i < len(lines) and lines[i].strip().startswith('_'):
            headers.append(lines[i].strip().split()[0])
            i += 1
        if not headers:
            raise ValueError(f'{source}: STAR loop_ has no column headers')
        hs = set(headers)
        has_coords = any((k in hs for k in required))
        if has_coords and (not all((k in hs for k in required))):
            missing = tuple((k for k in required if k not in hs))
            raise ValueError(f'{source}: incomplete CBOX coordinate loop; missing {missing}; headers={headers}')
        is_coords = all((k in hs for k in required))
        if is_coords:
            coordinate_loops += 1
            index = {key: headers.index(key) for key in required}
        while i < len(lines):
            s = lines[i].strip()
            if not s or s.startswith('#'):
                i += 1
                continue
            if s == 'loop_' or s.startswith('data_') or s.startswith('save_') or (s == 'stop_') or s.startswith('_'):
                break
            try:
                fields = shlex.split(s, comments=False, posix=True)
            except ValueError as exc:
                raise ValueError(f'{source}: malformed STAR row at line {i + 1}: {exc}') from exc
            if len(fields) != len(headers):
                raise ValueError(f'{source}: malformed STAR row at line {i + 1}: expected {len(headers)} fields, got {len(fields)}')
            if is_coords:
                coordinate_rows.append((index, fields, i + 1))
            i += 1
    if coordinate_loops == 0:
        raise ValueError(f'{source}: no CBOX coordinate loop found')
    return coordinate_rows

def cryolo_cbox_to_csv(cbox, outcsv, name, allow_empty=False):
    if not cbox.is_file():
        raise FileNotFoundError(f'crYOLO CBOX is missing: {cbox}')
    if cbox.stat().st_size == 0:
        if not allow_empty:
            raise ValueError(f'{cbox}: zero-byte CBOX is only valid when the successful crYOLO log confirms zero candidates')
        rows = []
    else:
        rows = parse_cryolo_cbox(cbox.read_text(encoding='utf-8'), str(cbox))
    out = []
    for index, row, line_number in rows:
        try:
            x = float(row[index['_CoordinateX']])
            y = float(row[index['_CoordinateY']])
            width = float(row[index['_Width']])
            height = float(row[index['_Height']])
            score = float(row[index['_Confidence']])
        except (ValueError, IndexError) as exc:
            raise ValueError(f'{cbox}: invalid coordinate/score at STAR line {line_number}: {exc}') from exc
        cx = x + width / 2.0
        cbox_cy = y + height / 2.0
        cy = IMAGE_HEIGHT - cbox_cy
        if not all(map(lambda z: z == z and abs(z) != float('inf'), (cx, cy, width, height, score))):
            raise ValueError(f'{cbox}: nonfinite coordinate/size/score at STAR line {line_number}')
        if width <= 0 or height <= 0:
            raise ValueError(f'{cbox}: nonpositive box dimensions at STAR line {line_number}: {width}x{height}')
        if not (0 <= cx < IMAGE_WIDTH and 0 <= cy < IMAGE_HEIGHT):
            raise ValueError(f'{cbox}: center outside original image at STAR line {line_number}: {cx},{cy}')
        out.append({'method': 'crYOLO', 'split': 'test', 'micrograph_name': name, 'x_original_px': cx, 'y_original_px': cy, 'score': score})
    with outcsv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'split', 'micrograph_name', 'x_original_px', 'y_original_px', 'score'])
        writer.writeheader()
        writer.writerows(out)
    return len(out)

def topaz_csv_to_canonical(source, outcsv, name):
    out = []
    with source.open(encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f, delimiter='\t')
        if not r.fieldnames or not {'x_coord', 'y_coord', 'score'}.issubset(r.fieldnames):
            f.seek(0)
            r = csv.DictReader(f)
        if not r.fieldnames or not {'x_coord', 'y_coord', 'score'}.issubset(r.fieldnames):
            raise ValueError(f'Unexpected Topaz output header: {r.fieldnames}')
        for i, row in enumerate(r):
            x = float(row['x_coord'])
            topaz_y = float(row['y_coord'])
            score = float(row['score'])
            if not all(map(lambda z: z == z and abs(z) != float('inf'), (x, topaz_y, score))):
                raise ValueError(f'{source}: nonfinite row {i}')
            if not (0 <= x < IMAGE_WIDTH and 0 <= topaz_y < IMAGE_HEIGHT):
                raise ValueError(f'{source}: coordinate outside Topaz up-scaled image: {x},{topaz_y}')
            y = IMAGE_HEIGHT - 1 - topaz_y
            if not 0 <= y < IMAGE_HEIGHT:
                raise ValueError(f'{source}: transformed coordinate outside original image: {x},{y}')
            out.append({'method': 'Topaz', 'split': 'test', 'micrograph_name': name, 'x_original_px': x, 'y_original_px': y, 'score': score})
    with outcsv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'split', 'micrograph_name', 'x_original_px', 'y_original_px', 'score'])
        w.writeheader()
        w.writerows(out)
    return len(out)