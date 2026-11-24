"""Browser playground for the optional local Chakkhu recognizer."""
from hashlib import sha256
from io import BytesIO
import json
import sys
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st

from tools import try_chakkhu as trial


@st.cache_resource
def reader():
    for name, checksum in trial.MODEL_FILES.items():
        trial.asset(trial.MODEL_DIR / name, f'{trial.MODEL_URL}/{name}', checksum, False)
    sys.path.insert(0, str(trial.MODEL_DIR))
    from chakkhu_onnx import Chakkhu
    return Chakkhu(str(trial.MODEL_DIR), threads=4)


@st.cache_resource
def line_finder():
    trial.asset(trial.DEMO_DIR / 'lines.py', f'{trial.DEMO_URL}/lines.py', trial.LINE_FINDER_HASH, False)
    import importlib.util
    spec = importlib.util.spec_from_file_location('chakkhu_trial_lines', trial.DEMO_DIR / 'lines.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_lines


def decode(data):
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError('Choose a nonempty JPG or PNG image up to 10 MB.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as source:
            if source.format not in {'JPEG', 'PNG'}:
                raise ValueError('Choose a JPG or PNG image.')
            if source.width * source.height > 20_000_000:
                raise ValueError('Choose an image with at most 20 million pixels.')
            source.verify()
        with Image.open(BytesIO(data)) as source:
            return ImageOps.exif_transpose(source).convert('RGB')


def render():
    st.set_page_config(page_title='Chakkhu OCR Test', page_icon='👁', layout='wide')
    st.title('Chakkhu OCR Test')
    st.caption('Read Khmer and English text from your own receipt using Chakkhu on your computer.')
    st.markdown('[Chakkhu by Ing Muyleang](https://huggingface.co/muyleanging/Chakkhu-OCR-Khmer-ONNX) · Apache-2.0')

    source = st.radio('Image source', ['Upload a receipt', 'Example receipt'], horizontal=True)
    data = None
    try:
        if source == 'Upload a receipt':
            upload = st.file_uploader('Choose your receipt', type=['jpg', 'jpeg', 'png'], max_upload_size=10)
            if upload is not None:
                data = upload.getvalue()
        else:
            trial.asset(trial.DEMO_DIR / 'receipt.jpg', f'{trial.DEMO_URL}/examples/docs/receipt.jpg',
                        trial.DEMO_IMAGE_HASH, False)
            data = (trial.DEMO_DIR / 'receipt.jpg').read_bytes()
        if data is None:
            st.session_state.pop('chakkhu_result', None)
            st.info('Upload a receipt to begin, or select Example receipt to try the public sample.')
            return
        image = decode(data)
    except Exception as exc:
        st.session_state.pop('chakkhu_result', None)
        st.error(f'Could not open image: {exc}')
        return

    signature = sha256(data).hexdigest()
    if signature != st.session_state.get('chakkhu_image'):
        st.session_state.pop('chakkhu_result', None)
        st.session_state.chakkhu_image = signature
    mode = st.radio('Read as', ['Full receipt', 'Single text line'], horizontal=True)
    quality = st.selectbox('Recognition mode', ['Best quality', 'Fast'])
    beam = 4 if quality == 'Best quality' else 0
    crop = st.checkbox('Crop image before scanning')
    bounds = (0, 0, image.width, image.height)
    if crop:
        horizontal = st.slider('Horizontal crop', 0, image.width, (0, image.width), key=f'x_{signature}')
        vertical = st.slider('Vertical crop', 0, image.height, (0, image.height), key=f'y_{signature}')
        bounds = (horizontal[0], vertical[0], horizontal[1], vertical[1])
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            st.warning('Expand the crop so it contains part of the image.')
            return
        image = image.crop(bounds)
    st.image(image, caption='Image to scan', width=420)
    if mode == 'Single text line':
        st.caption('Crop around one text line to test recognition without automatic line detection.')

    fingerprint = (signature, mode, beam, bounds)
    if st.button('Run Chakkhu OCR', type='primary'):
        st.session_state.pop('chakkhu_result', None)
        try:
            with st.spinner('Reading your image…'):
                model = reader()
                started = time.perf_counter()
                boxes = line_finder()(np.asarray(image)) if mode == 'Full receipt' else [(0, 0, image.width, image.height)]
                if not boxes or len(boxes) > 200:
                    raise ValueError('No usable lines or too many regions found. Crop the image and try again.')
                crops = [image.crop((x, y, x + w, y + h)) for x, y, w, h in boxes]
                predictions = model.read(crops, beam=beam, confidence=True, batch=4)
                rows = [{'Region': i + 1, 'Text': text, 'Score': score,
                         'Box': [int(value) for value in box]}
                        for i, (box, (text, score)) in enumerate(zip(boxes, predictions))]
                st.session_state.chakkhu_result = {
                    'fingerprint': fingerprint, 'rows': rows,
                    'text': '\n'.join(row['Text'] for row in rows if row['Text']),
                    'seconds': round(time.perf_counter() - started, 2),
                    'model_revision': trial.MODEL_REVISION,
                }
        except Exception as exc:
            st.error(f'Could not scan: {exc}')
            st.caption('If trial assets are missing, run: .\\.venv-chakkhu\\Scripts\\python tools\\try_chakkhu.py --download --demo')

    result = st.session_state.get('chakkhu_result')
    if result is None or result['fingerprint'] != fingerprint:
        return
    st.subheader('Recognized text')
    metrics = st.columns(2)
    metrics[0].metric('Text regions', sum(bool(row['Text']) for row in result['rows']))
    metrics[1].metric('Scan time', f"{result['seconds']:.2f} s")
    if not result['text']:
        st.warning('No text was recognized. Try a tighter crop or a clearer image.')
    st.code(result['text'], language=None)
    st.download_button('Download recognized text', result['text'], 'chakkhu-text.txt', mime='text/plain')
    report = {key: value for key, value in result.items() if key != 'fingerprint'}
    st.download_button('Download detailed results', json.dumps(report, ensure_ascii=False, indent=2),
                       'chakkhu-results.json', mime='application/json')
    st.caption('Images and results stay in this session. Use the download buttons to keep a copy.')

    with st.expander('Inspect detected regions', expanded=True):
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for row in result['rows']:
            x, y, w, h = row['Box']
            draw.rectangle((x, y, x + w, y + h), outline='#e63946', width=2)
            draw.text((x, max(0, y - 12)), str(row['Region']), fill='#e63946')
        st.image(overlay, caption='Check that boxes contain every digit and the complete text.', width=600)
        st.dataframe([{key: row[key] for key in ('Region', 'Text', 'Score')} for row in result['rows']],
                     hide_index=True, width='stretch')
        selected = st.selectbox('Inspect a region', [row['Region'] for row in result['rows']])
        x, y, w, h = result['rows'][selected - 1]['Box']
        st.image(image.crop((x, y, x + w, y + h)), caption=f'Region {selected}')
        st.caption('Scores are heuristic. A high score can still accompany missing or incorrect digits.')
    st.info('Check receipt totals against the image. This playground displays OCR text; saving expenses remains in SmartSpend.')


if __name__ == '__main__':
    render()
