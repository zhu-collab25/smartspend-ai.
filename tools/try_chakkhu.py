"""Local OCR trial: Chakkhu by Ing Muyleang, Apache-2.0.

Downloads are explicit and pinned. Receipt images are never uploaded. This is a
recognition experiment, independent of the app's English receipt parser/storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = '8d7373487dfb56e4f13ba26cea31e1f858bd6ddd'
DEMO_REVISION = 'd2e2981bd7df5e9ec1f149ce713dba7d8b9e1a3b'
MODEL_DIR = ROOT / '.cache' / 'chakkhu' / MODEL_REVISION
DEMO_DIR = ROOT / '.cache' / 'chakkhu' / 'demo'
MODEL_URL = f'https://huggingface.co/muyleanging/Chakkhu-OCR-Khmer-ONNX/resolve/{MODEL_REVISION}'
DEMO_URL = f'https://huggingface.co/spaces/muyleanging/Chakkhu-OCR-Demo/resolve/{DEMO_REVISION}'
MODEL_FILES = {
    'chakkhu_onnx.py': 'c5054cb6ab9cb70a647229d95b085da7c4ff724191b295a86aa3f0bff99029ee',
    'khnormal.py': '127591c2e33ef6f37937801730df494e787d77f5d56ddd49f98d5cdbba75d151',
    'tokenizer.json': '42dad64377998f2dc5176553c5bdc4cd7384e826c9e9cf554aa955e6a861fe58',
    'encoder.int8.onnx': '00a304e215c27ac53d7b2ed4660c6fd37b6815f6b1e8cff599c713556992cc8a',
    'decoder.int8.onnx': 'acb54fcb4ff9f1fd98e71f38dc8dcef489d7cf6397aa2bdc9d9b4cb1db5e5ebf',
}
LINE_FINDER_HASH = 'd89a4233b68bd7e948f03593917fcd1128a37122b357812c28b03bbdbbc475df'
DEMO_IMAGE_HASH = 'b106b57879aacd1458c3c0c43d16a6f544ae1a494e4a3c9c508f6657021075bc'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def asset(path, url, expected, download):
    if path.is_file():
        if digest(path) != expected:
            raise ValueError(f'Checksum mismatch: {path}. Remove this file and download it again.')
        return
    if not download:
        raise ValueError('Trial assets are missing. Run with --download once while online.')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    try:
        print(f'Downloading {path.name}...', file=sys.stderr, flush=True)
        with urlopen(url, timeout=120) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
        if digest(partial) != expected:
            raise ValueError(f'Download checksum mismatch: {path.name}')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, nargs='?', help='One text-line image; use --page for a full receipt.')
    parser.add_argument('--download', action='store_true', help='Download pinned public assets if missing.')
    parser.add_argument('--demo', action='store_true', help="Read the author's public Khmer demo receipt.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--page', action='store_true', help="Use the author's experimental line finder first.")
    mode.add_argument('--crop', type=int, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    parser.add_argument('--beam', type=int, choices=[0, 4], default=4, help='4: beam + CTC; 0: faster CTC only.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, help='Explicitly save extracted text, boxes, and timings as JSON.')
    args = parser.parse_args()
    if args.demo and args.image:
        parser.error('Choose either an image path or --demo.')
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if not (args.image or args.demo or args.download):
        parser.error('Supply an image, --demo, or --download.')

    for name, checksum in MODEL_FILES.items():
        asset(MODEL_DIR / name, f'{MODEL_URL}/{name}', checksum, args.download)
    if args.page or args.demo:
        asset(DEMO_DIR / 'lines.py', f'{DEMO_URL}/lines.py', LINE_FINDER_HASH, args.download)
    if args.demo:
        asset(DEMO_DIR / 'receipt.jpg', f'{DEMO_URL}/examples/docs/receipt.jpg', DEMO_IMAGE_HASH, args.download)
    if not (args.image or args.demo):
        print('Model ready. Use --demo --download or provide a text-line image.')
        return

    # These two upstream modules are inspected, revision-pinned, and verified above.
    sys.path.insert(0, str(MODEL_DIR))
    from chakkhu_onnx import Chakkhu
    import numpy as np
    from PIL import Image, ImageOps

    image_path = DEMO_DIR / 'receipt.jpg' if args.demo else args.image
    with Image.open(image_path) as source:
        if source.width * source.height > 20_000_000:
            raise ValueError('Use an image of at most 20 million pixels.')
        image = ImageOps.exif_transpose(source).convert('RGB')
    original_size = image.size
    if args.crop:
        left, top, right, bottom = args.crop
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError('Crop coordinates must be inside the image.')
        image = image.crop(tuple(args.crop))

    page = args.page or (args.demo and not args.crop)
    started = time.perf_counter()
    if page:
        sys.path.insert(0, str(DEMO_DIR))
        from lines import find_lines
        boxes = find_lines(np.asarray(image))
    else:
        boxes = [(0, 0, image.width, image.height)]
    detector_seconds = time.perf_counter() - started
    if not boxes or len(boxes) > 200:
        raise ValueError('No usable lines or too many lines detected. Try cropping the receipt.')
    crops = [image.crop((x, y, x + w, y + h)) for x, y, w, h in boxes]
    started = time.perf_counter()
    reader = Chakkhu(str(MODEL_DIR), threads=args.threads)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    recognized = reader.read(crops, beam=args.beam, confidence=True, batch=4)
    read_seconds = time.perf_counter() - started
    report = {
        'model': 'muyleanging/Chakkhu-OCR-Khmer-ONNX',
        'model_revision': MODEL_REVISION,
        'line_finder_revision': DEMO_REVISION if page else None,
        'input': str(image_path), 'original_size': original_size, 'crop': args.crop,
        'mode': 'page' if page else 'line', 'beam': args.beam, 'threads': args.threads,
        'provider': 'CPUExecutionProvider', 'load_seconds': round(load_seconds, 3),
        'detector_seconds': round(detector_seconds, 3), 'read_seconds': round(read_seconds, 3),
        'confidence_note': 'Heuristic line score; not a calibrated probability of correctness.',
        'lines': [{'box_xywh': [int(v) for v in box], 'text': text, 'confidence': score}
                  for box, (text, score) in zip(boxes, recognized)],
        'text': '\n'.join(text for text, _ in recognized),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # ASCII escapes also work on Windows terminals without Khmer font/UTF-8 support.
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
