"""Pinned, local Chakkhu OCR adapter for receipt scanning."""
from functools import lru_cache
import importlib.util
import sys
from threading import Lock
from time import monotonic

from tools import try_chakkhu as assets

_INFERENCE_LOCK = Lock()
SETUP = 'python tools/try_chakkhu.py --download --page'


class ChakkhuError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_reader():
    for name, checksum in assets.MODEL_FILES.items():
        assets.asset(assets.MODEL_DIR / name, f'{assets.MODEL_URL}/{name}', checksum, False)
    sys.path.insert(0, str(assets.MODEL_DIR))
    from chakkhu_onnx import Chakkhu
    return Chakkhu(str(assets.MODEL_DIR), threads=4)


@lru_cache(maxsize=1)
def load_line_finder():
    assets.asset(assets.DEMO_DIR / 'lines.py', f'{assets.DEMO_URL}/lines.py', assets.LINE_FINDER_HASH, False)
    spec = importlib.util.spec_from_file_location('chakkhu_receipt_lines', assets.DEMO_DIR / 'lines.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_lines


def layout_text(regions):
    """Join separate label/price regions on the same visual row, left to right."""
    rows = []
    for region in sorted(regions, key=lambda r: r['box'][1] + r['box'][3] / 2):
        if not region['text'].strip():
            continue
        x, y, w, h = region['box']
        center = y + h / 2
        for row in rows:
            overlaps = any(min(x + w, p['box'][0] + p['box'][2]) - max(x, p['box'][0])
                           > 0.2 * min(w, p['box'][2]) for p in row['parts'])
            if not overlaps and abs(center - row['center']) <= 0.7 * max(h, row['height']):
                row['parts'].append(region)
                break
        else:
            rows.append({'center': center, 'height': h, 'parts': [region]})
    return '\n'.join(' '.join(p['text'].strip() for p in sorted(row['parts'], key=lambda r: r['box'][0]))
                     for row in rows)


def recognize_receipt(image):
    """Read in memory; return assembled text plus source regions for review."""
    try:
        import numpy as np
        model, find_lines = load_reader(), load_line_finder()
    except ImportError as exc:
        raise ChakkhuError('Chakkhu dependencies are missing. Install requirements.txt, or choose Tesseract.') from exc
    except (OSError, ValueError) as exc:
        raise ChakkhuError(f'Chakkhu model files are missing or damaged. Run {SETUP}, or choose Tesseract.') from exc
    with _INFERENCE_LOCK:
        started = monotonic()
        boxes = find_lines(np.asarray(image.convert('RGB')))
        if not boxes or len(boxes) > 200:
            raise ChakkhuError('No usable text regions or too many regions found. Try a cropped image or enter details manually.')
        regions = []
        for offset in range(0, len(boxes), 4):
            if monotonic() - started > 45:
                raise ChakkhuError('Scanning took too long. Try a shorter, cropped receipt or enter details manually.')
            part = boxes[offset:offset + 4]
            crops = [image.crop((x, y, x + w, y + h)) for x, y, w, h in part]
            predictions = model.read(crops, beam=4, confidence=True, batch=4)
            regions.extend({'box': [int(v) for v in box], 'text': text, 'score': score}
                           for box, (text, score) in zip(part, predictions))
        if monotonic() - started > 45:
            raise ChakkhuError('Scanning took too long. Try a shorter, cropped receipt or enter details manually.')
    text = layout_text(regions)
    if not text.strip():
        raise ChakkhuError('No readable text found. Try a clearer photo or enter details manually.')
    return text, regions
