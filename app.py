"""
app.py — Streamlit frontend for the rolling ticker extraction pipeline.

Wraps the existing pipeline modules without modifying any of them.
Two modes:
  Full length — seek to a timestamp, read the ticker to the end of the video
  Segment     — split the video on scene changes first, read each segment

Run with:  streamlit run app.py
"""

import os
import sys
import re
import html
import types
import base64
import tempfile

# Pipeline modules use flat sibling imports (e.g. `from video import Video`)
# rather than a proper package, so extend sys.path instead of rewriting every
# import across pipeline/*.py. Nothing inside pipeline/ needed to change.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline'))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

MASTER_DIR = './samples/master_files'
os.makedirs(MASTER_DIR, exist_ok=True)

BASE_PARAMETERS = {
    'jump_size'        : 12,
    'resize_font'      : True,
    'height'           : None,
    'new_height'       : 32,
    'interpolation'    : 'cubic',
    'add_padding'      : False,
    'method'           : None,
    'gamma_correct'    : True,
    'merge_method'     : 'confidence',
    'garbage_method'   : 'char_confidence',
    'garbage_threshold': 0.80,
    'overlap_method'   : 'merges',
    'video_id'         : 'unknown',
    'master_tsv_path'  : f'{MASTER_DIR}/master_words.tsv',
}

SEG_METHOD     = 'Bhattachrya Distance'
SEG_THRESHOLD  = 0.7
MIN_SEG_SECS   = 3.0
TARGET_SAMPLES = 80

# The synced player embeds the clip as a base64 data URI (see build_sync_panel).
# Base64 inflates the file by ~33%, so a 45 MB clip lands near 60 MB of markup.
# That is fine for the short broadcast clips this tool is used on. Above this
# ceiling the browser starts to choke, so the panel drops the video and shows
# the transcript alone. If full-length recordings ever need to play here, the
# fix is static file serving (.streamlit/config.toml → enableStaticServing)
# rather than a bigger data URI — deliberately not built, it is not needed yet.
MAX_EMBED_BYTES = 45 * 1024 * 1024

MIME_BY_EXT = {
    '.mp4' : 'video/mp4',
    '.webm': 'video/webm',
    '.mov' : 'video/quicktime',
    '.avi' : 'video/x-msvideo',
}


def _full_width():
    """Streamlit renamed use_container_width to width in 1.49 — support both."""
    try:
        major, minor = (int(p) for p in st.__version__.split('.')[:2])
        if (major, minor) >= (1, 49):
            return {'width': 'stretch'}
    except (ValueError, AttributeError):
        pass
    return {'use_container_width': True}


FULL_WIDTH = _full_width()


# ─────────────────────────────────────────────────────────────────────
# Shared text helpers (same rules as easyrun.py / run_on_segments.py)
# ─────────────────────────────────────────────────────────────────────

def parse_timestamp(ts: str) -> float:
    parts = ts.strip().split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def clean_story(text: str) -> str:
    words = text.split()
    if not words:
        return text
    result = [words[0]]
    for word in words[1:]:
        if word.lower() != result[-1].lower():
            result.append(word)
    return ' '.join(result).strip()


def is_valid_story(text: str) -> bool:
    words = text.split()
    if len(words) < 3:
        return False
    if len([w for w in words if len(w) >= 3]) < 2:
        return False
    if len(text) < 15:
        return False
    return True


def clean_output(text: str) -> str:
    text = re.sub(r'[\{\[\(]', '', text)
    text = re.sub(r'\|+', '|', text)
    text = re.sub(r'\s*\|\s*', ' | ', text)
    text = re.sub(r'^\s*\|\s*', '', text)
    text = re.sub(r'\s*\|\s*$', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def percentages_to_window(left_pct, top_pct, width_pct, height_pct,
                          frame_width, frame_height):
    """Returns (ticker_window, t_height, t_width) in the pipeline's format."""
    minx = int(left_pct * frame_width)
    miny = int(top_pct  * frame_height)
    maxx = int((left_pct + width_pct)  * frame_width)  - 1
    maxy = int((top_pct  + height_pct) * frame_height) - 1
    ticker_window = [[miny, maxy], [minx, maxx]]
    return ticker_window, maxy - miny + 1, maxx - minx + 1


def probe_video(video_path):
    """Frame size, fps and duration without walking the whole file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fw, fh, fps, (count / fps if fps else 0.0)


def crop_preview(video_path, ticker_window, at_fraction=0.10):
    """Frame at `at_fraction` of the video, cropped to the ticker window."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * at_fraction)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    (miny, maxy), (minx, maxx) = ticker_window
    crop = frame[miny:maxy + 1, minx:maxx + 1]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────
# Pipeline calls
# ─────────────────────────────────────────────────────────────────────

def run_full_extraction(video_path, timestamp_str, ticker_window,
                        t_height, t_width, script_path, best_parameters):
    from video import Video
    from slidingreader import read_ticker, add_story_start_end_times
    from helpers import concatenate_news_stories
    from tesseract import TesseractOCR
    from evaluator import evaluate_all

    seek_seconds = parse_timestamp(timestamp_str)
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    params = dict(best_parameters)
    params['video_id'] = video_id
    params['height']   = t_height
    params['width']    = t_width

    video = Video(video_path, language='English')
    video.seek(seek_seconds)

    ocr = TesseractOCR(**params)
    ocr._preprocesses['height'] = t_height

    raw_stories = read_ticker(video, ticker_window, ocr, **params)
    add_story_start_end_times(raw_stories, video)

    jump = params.get('jump_size', 12) or 1
    for s in raw_stories:
        s['frequency'] = max(1, (s.get('maxframe', 0) - s.get('minframe', 0)) // jump)

    try:
        video.video_capture.release()
    except Exception:
        pass

    for s in raw_stories:
        s['text'] = clean_story(s['text'])
    stories = [s for s in raw_stories if is_valid_story(s['text'])]

    output = clean_output(concatenate_news_stories(stories, char=' | '))

    for s in stories:
        s['start_ts'] = seconds_to_hms(s.get('start_time', 0))
        s['end_ts']   = seconds_to_hms(s.get('end_time', 0))

    story_frequencies = [s.get('frequency') for s in stories]

    eval_results = None
    if script_path and os.path.exists(script_path):
        eval_results = evaluate_all(output, script_path,
                                    frequencies=story_frequencies)

    return {
        'output'       : output,
        'stories'      : stories,
        'eval_results' : eval_results,
        'story_freqs'  : story_frequencies,
        'video_id'     : video_id,
    }


def run_segment_extraction(video_path, ticker_window, t_height, t_width,
                           script_path, best_parameters, progress_cb=None):
    from video import Video
    from video_segmentor import VideoSegmentation
    from slidingreader import read_ticker, add_story_start_end_times
    from helpers import concatenate_news_stories
    from tesseract import TesseractOCR
    from evaluator import evaluate_segments

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    seg = VideoSegmentation(path=video_path, no_of_bins=16,
                            frame_skip=28, threshold=SEG_THRESHOLD)
    _, fps, _, duration, _, segment_time = seg.segment_video(method=SEG_METHOD)

    boundaries = [0.0] + list(segment_time) + [duration]
    segments   = [(boundaries[i], boundaries[i + 1])
                  for i in range(len(boundaries) - 1)]

    total_segments = len(segments)
    valid_segments = sum(1 for s, e in segments if e - s >= MIN_SEG_SECS) - 1

    results = []
    for i, (start, end) in enumerate(segments):
        if progress_cb:
            progress_cb(i + 1, total_segments)

        seg_dur = end - start
        if seg_dur < MIN_SEG_SECS:
            continue

        seg_frame_count = int(seg_dur * fps)
        dynamic_jump    = max(1, seg_frame_count // TARGET_SAMPLES)

        params = dict(best_parameters)
        params['jump_size'] = dynamic_jump
        params['height']    = t_height
        params['width']     = t_width
        params['video_id']  = video_id

        try:
            video = Video(video_path, language='English')
            video.seek(start)

            seek_frame = int(video.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            video.frame_count = seg_frame_count

            original_frame_fn = video.frame.__func__

            def offset_frame(self, frame_number=None):
                if frame_number is not None:
                    frame_number += self._seek_offset
                return original_frame_fn(self, frame_number)

            video._seek_offset = seek_frame
            video.frame = types.MethodType(offset_frame, video)

            ocr = TesseractOCR(**params)
            ocr._preprocesses['height'] = t_height

            raw_stories = read_ticker(video, ticker_window, ocr, **params)
            add_story_start_end_times(raw_stories, video)

            try:
                video.video_capture.release()
            except Exception:
                pass

            for story in raw_stories:
                story['text'] = clean_story(story['text'])

            stories = [s for s in raw_stories if is_valid_story(s['text'])]
            if not stories:
                continue

            output = clean_output(
                concatenate_news_stories(stories, char=' | '))

            frag_cer = frag_wer = None
            per_story_eval = []
            if script_path and os.path.exists(script_path):
                seg_eval = evaluate_segments(output, script_path)
                frag_cer = seg_eval['fragment_cer']
                frag_wer = seg_eval['fragment_wer']
                per_story_eval = seg_eval['per_story']

            results.append({
                'segment'      : i + 1,
                'start'        : seconds_to_hms(start),
                'end'          : seconds_to_hms(end),
                # Raw seconds are what the synced player highlights on; the
                # HH:MM:SS strings above stay for display and CSV export.
                'start_sec'    : round(float(start), 3),
                'end_sec'      : round(float(end), 3),
                'duration'     : round(seg_dur, 2),
                'jump_size'    : dynamic_jump,
                'stories'      : len(stories),
                'extracted'    : output,
                'fragment_cer' : frag_cer,
                'fragment_wer' : frag_wer,
                'per_story'    : per_story_eval,
            })

        except Exception:
            continue

    valid_results = [r for r in results
                     if isinstance(r.get('fragment_cer'), float)]
    avg_cer = avg_wer = None
    if valid_results:
        avg_cer = round(sum(r['fragment_cer'] for r in valid_results)
                        / len(valid_results), 4)
        avg_wer = round(sum(r['fragment_wer'] for r in valid_results)
                        / len(valid_results), 4)

    return {
        'total_segments' : total_segments,
        'valid_segments' : valid_segments,
        'segments'       : results,
        'avg_frag_cer'   : avg_cer,
        'avg_frag_wer'   : avg_wer,
        'video_id'       : video_id,
    }


# ─────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────

def full_results_to_csv(results) -> str:
    per_story = (results['eval_results']['per_story']
                 if results.get('eval_results') else [])
    rows = []
    for i, story in enumerate(results['stories']):
        ev = per_story[i] if i < len(per_story) else {}
        rows.append({
            'story_index'      : i + 1,
            'start_ts'         : story.get('start_ts', ''),
            'end_ts'           : story.get('end_ts', ''),
            'extracted_text'   : story.get('text', ''),
            'frequency'        : story.get('frequency', ''),
            'cer'              : ev.get('cer', ''),
            'wer'              : ev.get('wer', ''),
            'matched_reference': ev.get('matched_reference', ''),
            'similarity'       : ev.get('similarity', ''),
        })
    return pd.DataFrame(rows).to_csv(index=False)


def segment_results_to_csv(results) -> str:
    rows = [{
        'segment'      : r['segment'],
        'start'        : r['start'],
        'end'          : r['end'],
        'duration'     : r['duration'],
        'jump_size'    : r['jump_size'],
        'stories'      : r['stories'],
        'extracted'    : r['extracted'],
        'fragment_cer' : r['fragment_cer'] if r['fragment_cer'] is not None else 'N/A',
        'fragment_wer' : r['fragment_wer'] if r['fragment_wer'] is not None else 'N/A',
    } for r in results['segments']]
    return pd.DataFrame(rows).to_csv(index=False)


# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────

CSS = """
<style>
  .stApp { background: #0d0f13; }
  section.main > div { padding-top: 1rem; }
  h1, h2, h3, h4 { color: #e8eaed !important; letter-spacing: -0.01em; }
  .tk-header {
      display: flex; align-items: center; gap: .7rem;
      padding: .9rem 1.2rem; margin-bottom: 1.2rem;
      background: #14171d; border: 1px solid #232833; border-radius: 10px;
  }
  .tk-header .tk-mark {
      width: 30px; height: 30px; border-radius: 7px;
      background: linear-gradient(135deg,#2f6fed,#1b4bb8);
      display: flex; align-items: center; justify-content: center;
      font-size: 15px;
  }
  .tk-header .tk-title {
      font-size: 1.15rem; font-weight: 600; color: #e8eaed;
  }
  .tk-panel-label {
      font-size: .72rem; font-weight: 600; letter-spacing: .12em;
      text-transform: uppercase; color: #7d8694; margin-bottom: .6rem;
  }
  .tk-empty {
      border: 1px dashed #2b3140; border-radius: 10px;
      padding: 3rem 1.5rem; text-align: center;
      color: #6b7280; font-size: .92rem; background: #11141a;
  }
  div[data-testid="stExpander"] {
      border: 1px solid #232833 !important; border-radius: 9px !important;
      background: #14171d !important; margin-bottom: .5rem;
  }
  div[data-testid="stMetric"] {
      background: #14171d; border: 1px solid #232833;
      border-radius: 9px; padding: .7rem .9rem;
  }
  div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
  .tk-story {
      background: #10131a; border-left: 2px solid #2f6fed;
      padding: .7rem .9rem; border-radius: 4px;
      color: #dfe3e8; font-size: .95rem; line-height: 1.55;
      margin-bottom: .8rem;
  }

  /* ── Extraction overlay ────────────────────────────────────────────
     Streamlit strips <script> from st.markdown, so the elapsed counter
     is two pure-CSS digit reels: a strip of numbers scrolled one line at
     a time by a steps() animation behind a one-line-tall clip window.
     Seconds tick every 1s and loop; minutes tick every 60s and hold at
     59:59. Nothing here needs Python, so it keeps animating while the
     pipeline blocks the Streamlit script. */
  .tk-overlay {
      position: fixed; inset: 0; z-index: 99999;
      background: rgba(0, 0, 0, .72);
      backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
      display: flex; align-items: center; justify-content: center;
      animation: tk-fade .25s ease;
  }
  @keyframes tk-fade { from { opacity: 0 } to { opacity: 1 } }
  .tk-overlay-card {
      width: 380px; padding: 2.5rem 2rem;
      background: #14171d; border: 1px solid #2f6fed; border-radius: 14px;
      display: flex; flex-direction: column; align-items: center;
      text-align: center;
      box-shadow: 0 0 40px rgba(47,111,237,.22), 0 24px 50px rgba(0,0,0,.7);
  }
  .tk-overlay-spinner {
      width: 44px; height: 44px; margin-bottom: 1.4rem; border-radius: 50%;
      border: 3px solid rgba(96,165,250,.22); border-top-color: #60a5fa;
      animation: tk-spin .9s linear infinite;
  }
  @keyframes tk-spin { to { transform: rotate(360deg) } }
  .tk-overlay-title {
      font-size: 1.05rem; font-weight: 600; color: #e8eaed;
      margin-bottom: .2rem;
  }
  .tk-overlay-sub {
      font-size: .82rem; color: #7d8694; margin-bottom: 1.3rem;
  }
  .tk-elapsed {
      display: flex; align-items: center; gap: .45rem;
      font-size: 1.15rem; font-weight: 500; color: #60a5fa;
  }
  .tk-elapsed-label { font-size: .9rem; color: #a0a0a0; }
  .tk-reel {
      display: inline-block; height: 1.5rem; overflow: hidden;
      font-variant-numeric: tabular-nums;
  }
  .tk-reel i { display: block; height: 1.5rem; line-height: 1.5rem;
               font-style: normal; }
  .tk-strip { display: block; }
  .tk-reel-min .tk-strip { animation: tk-roll 3600s steps(60, end) forwards; }
  .tk-reel-sec .tk-strip { animation: tk-roll   60s steps(60, end) infinite; }
  @keyframes tk-roll {
      from { transform: translateY(0); }
      to   { transform: translateY(-90rem); }   /* 60 lines × 1.5rem */
  }
  @media (prefers-reduced-motion: reduce) {
      .tk-overlay-spinner { animation-duration: 2.4s; }
  }
</style>
"""


# ─────────────────────────────────────────────────────────────────────
# Extraction overlay (Feature 1)
# ─────────────────────────────────────────────────────────────────────

def show_extraction_overlay(subtitle: str, slot=None):
    """Blur the page and start a client-side elapsed counter.

    Called immediately before the blocking pipeline call. Streamlit flushes
    this delta to the browser before Python blocks, so the overlay is up and
    animating for the whole run and vanishes on the next st.rerun().

    `slot` is an st.empty() created at the top level of the page. Rendering
    into it keeps the overlay a direct child of the app root rather than of a
    column, so nothing upstream can turn into a containing block and trap the
    position:fixed layer inside one panel.
    """
    # 61 minute cells: the steps() animation lands on cell 60 when it
    # completes, so the trailing repeat holds the reel at 59 past an hour
    # instead of scrolling past the end into blank space.
    minutes = ''.join(f'<i>{m}</i>' for m in list(range(60)) + [59])
    seconds = ''.join(f'<i>{s:02d}</i>' for s in range(60))
    (slot or st).markdown(
        f'''
        <div class="tk-overlay">
          <div class="tk-overlay-card">
            <div class="tk-overlay-spinner"></div>
            <div class="tk-overlay-title">Extracting ticker text…</div>
            <div class="tk-overlay-sub">{html.escape(subtitle)}</div>
            <div class="tk-elapsed">
              <span class="tk-elapsed-label">Elapsed Time:</span>
              <span class="tk-reel tk-reel-min"><span class="tk-strip">{minutes}</span></span>
              <span>:</span>
              <span class="tk-reel tk-reel-sec"><span class="tk-strip">{seconds}</span></span>
            </div>
          </div>
        </div>
        ''',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Synced video + transcript panel (Features 2 and 3)
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=2)
def video_data_uri(video_path: str, size: int, mtime: float):
    """Base64 data URI for the clip, or None if it is too large to embed.

    `size` and `mtime` are cache keys only — a new upload lands on a new
    temp path anyway, but this keeps the encode off every widget rerun.
    """
    if size > MAX_EMBED_BYTES:
        return None
    ext  = os.path.splitext(video_path)[1].lower()
    mime = MIME_BY_EXT.get(ext, 'video/mp4')
    with open(video_path, 'rb') as fh:
        encoded = base64.b64encode(fh.read()).decode('ascii')
    return f"data:{mime};base64,{encoded}"


def _metric_chips(pairs):
    """Small inline stat chips, e.g. CER 0.12 · WER 0.30."""
    chips = ''.join(
        f'<span class="chip"><em>{html.escape(str(k))}</em>'
        f'{html.escape(str(v))}</span>'
        for k, v in pairs if v is not None)
    return f'<div class="chips">{chips}</div>' if chips else ''


def _word_error_html(word_errors):
    if not word_errors:
        return ''
    rows = []
    for err in word_errors:
        if err['type'] == 'SUB':
            rows.append(f'<li class="sub">SUB &nbsp;'
                        f'<s>{html.escape(str(err["expected"]))}</s> → '
                        f'<b>{html.escape(str(err["got"]))}</b></li>')
        elif err['type'] == 'DEL':
            rows.append(f'<li class="del">DEL &nbsp;'
                        f'<b>{html.escape(str(err["expected"]))}</b> missing</li>')
        elif err['type'] == 'INS':
            rows.append(f'<li class="ins">INS &nbsp;'
                        f'<b>{html.escape(str(err["got"]))}</b> hallucinated</li>')
    return ('<div class="errs-label">Word errors</div>'
            f'<ul class="errs">{"".join(rows)}</ul>')


def full_entries(results):
    """Transcript entries for Full Length mode.

    start_time / end_time are already in seconds on each story, filled in by
    add_story_start_end_times, so they map straight onto playback position.
    """
    per_story = (results['eval_results']['per_story']
                 if results.get('eval_results') else [])
    entries = []
    for i, story in enumerate(results['stories']):
        start = float(story.get('start_time') or 0.0)
        end   = float(story.get('end_time') or 0.0)
        if end <= start:
            end = start + 2.0

        body = (f'<div class="text">{html.escape(story.get("text", ""))}</div>')

        if i < len(per_story):
            s = per_story[i]
            body += _metric_chips([('CER', s.get('cer')),
                                   ('WER', s.get('wer')),
                                   ('sim', f'{s.get("similarity")}%'
                                    if s.get('similarity') is not None else None)])
            if s.get('matched_reference'):
                body += (f'<div class="ref"><em>Reference</em>'
                         f'{html.escape(str(s["matched_reference"]))}</div>')
            body += _word_error_html(s.get('word_errors'))

        body += (f'<div class="foot">Visible across ~'
                 f'{story.get("frequency", 0)} sampled frames</div>')

        entries.append({
            'start': start,
            'end'  : end,
            'label': f'Story {i + 1}',
            'time' : f'{story.get("start_ts", "--:--:--")} → '
                     f'{story.get("end_ts", "--:--:--")}',
            'body' : body,
        })
    return entries


def segment_entries(results):
    """Transcript entries for Segment mode.

    evaluate_segments returns per-story dicts with no timestamps of their own,
    so highlighting stays at the segment level — the whole block lights up
    while playback sits inside [start_sec, end_sec).
    """
    entries = []
    for seg in results['segments']:
        body = f'<div class="text">{html.escape(seg.get("extracted", ""))}</div>'

        if seg.get('fragment_cer') is not None:
            body += _metric_chips([('Fragment CER', seg['fragment_cer']),
                                   ('Fragment WER', seg['fragment_wer'])])

        for s in seg.get('per_story', []):
            body += (
                '<div class="sub-story">'
                f'<div class="sub-head">Story {s["story_index"]} &nbsp;·&nbsp; '
                f'CER {s["fragment_cer"]} &nbsp;·&nbsp; WER {s["fragment_wer"]} '
                f'&nbsp;·&nbsp; sim {s["similarity"]}%</div>'
                f'<div class="sub-line"><em>Extracted</em>'
                f'{html.escape(str(s.get("extracted", "")))}</div>'
                f'<div class="sub-line"><em>Aligned</em>'
                f'{html.escape(str(s.get("aligned_reference", "")))}</div>'
                '</div>')

        body += (f'<div class="foot">jump_size {seg["jump_size"]} · '
                 f'{seg["stories"]} stories · {seg["duration"]}s</div>')

        entries.append({
            'start': float(seg.get('start_sec') or 0.0),
            'end'  : float(seg.get('end_sec') or 0.0),
            'label': f'Segment {seg["segment"]}',
            'time' : f'{seg["start"]} → {seg["end"]}',
            'body' : body,
        })
    return entries


SYNC_PANEL_TEMPLATE = """
<style>
  :root {
      --bg-dark:#0d0f13; --bg-panel:#14171d; --bg-card:#1a1d25;
      --bg-active:#18243a; --border:#232833; --border-active:#2f6fed;
      --text:#e8eaed; --text-2:#a0a6b0; --muted:#6b7280; --accent:#60a5fa;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html { background:transparent; }
  body {
      background:transparent; color:var(--text);
      font-family:'Source Sans Pro', -apple-system, BlinkMacSystemFont,
                  'Segoe UI', sans-serif;
      -webkit-font-smoothing:antialiased;
  }
  .wrap { display:flex; flex-direction:column; gap:.75rem; height:__HEIGHT__px; }

  /* Controls sit in their own row ABOVE the frame. Native controls are off:
     on pause the browser paints them as a bar across the bottom of the video,
     which is exactly where the ticker lives (top ≈ 0.90 of frame height). */
  .bar {
      display:flex; align-items:center; gap:.7rem;
      background:var(--bg-panel); border:1px solid var(--border);
      border-radius:8px; padding:.5rem .7rem; flex:0 0 auto;
  }
  .bar button {
      width:30px; height:30px; flex:0 0 auto; border-radius:6px;
      border:1px solid var(--border); background:#1f2430; color:var(--text);
      cursor:pointer; font-size:12px; line-height:1;
      display:flex; align-items:center; justify-content:center;
  }
  .bar button:hover { border-color:var(--border-active); color:#fff; }
  .bar button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .bar input[type=range] {
      flex:1 1 auto; appearance:none; height:4px; border-radius:3px;
      background:#2b3140; cursor:pointer;
  }
  .bar input[type=range]::-webkit-slider-thumb {
      appearance:none; width:13px; height:13px; border-radius:50%;
      background:var(--border-active); border:2px solid #0d0f13;
  }
  .bar input[type=range]::-moz-range-thumb {
      width:13px; height:13px; border-radius:50%; border:2px solid #0d0f13;
      background:var(--border-active);
  }
  .clock {
      font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size:.75rem; color:var(--text-2); flex:0 0 auto;
  }

  .stage { background:#000; border-radius:8px; overflow:hidden; flex:0 0 auto; }
  video { display:block; width:100%; max-height:__VIDEOH__px; background:#000; }
  .novideo {
      padding:1.6rem; text-align:center; color:var(--muted);
      font-size:.85rem; background:var(--bg-panel);
      border:1px dashed var(--border); border-radius:8px;
  }

  .list { flex:1 1 auto; overflow-y:auto; padding-right:.35rem;
          display:flex; flex-direction:column; gap:.5rem; }
  .list::-webkit-scrollbar { width:7px; }
  .list::-webkit-scrollbar-thumb { background:#2b3140; border-radius:4px; }

  .entry {
      background:var(--bg-card); border:1px solid var(--border);
      border-radius:9px; padding:.8rem .9rem; cursor:pointer;
      transition:background .18s, border-color .18s, box-shadow .18s;
  }
  .entry:hover { background:#22262f; }
  .entry.tk-active {
      background:var(--bg-active); border-color:var(--border-active);
      box-shadow:0 0 0 1px rgba(47,111,237,.35), 0 0 18px rgba(47,111,237,.18);
  }
  .head {
      display:flex; justify-content:space-between; align-items:center;
      gap:.6rem; margin-bottom:.5rem;
  }
  .name { font-size:.8rem; font-weight:600; color:var(--text-2); }
  .tk-active .name { color:var(--accent); }
  .stamp {
      font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size:.72rem; color:var(--muted);
  }
  .text {
      border-left:2px solid #2f6fed; background:#10131a;
      padding:.55rem .7rem; border-radius:4px;
      font-size:.92rem; line-height:1.55; color:#dfe3e8;
      word-break:break-word;
  }
  .chips { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.55rem; }
  .chip {
      font-size:.72rem; color:var(--text-2); background:#12151c;
      border:1px solid var(--border); border-radius:5px; padding:.2rem .45rem;
  }
  .chip em { font-style:normal; color:var(--muted); margin-right:.35rem; }
  .ref, .sub-line { font-size:.8rem; color:var(--text-2);
                    margin-top:.5rem; line-height:1.5; }
  .ref em, .sub-line em {
      font-style:normal; color:var(--muted); margin-right:.4rem;
      text-transform:uppercase; font-size:.66rem; letter-spacing:.08em;
  }
  .errs-label { font-size:.72rem; color:var(--muted); margin-top:.6rem;
                text-transform:uppercase; letter-spacing:.08em; }
  .errs { list-style:none; margin-top:.3rem; display:flex;
          flex-direction:column; gap:.22rem; }
  .errs li {
      font-size:.78rem; padding:.24rem .45rem; border-radius:4px;
      border-left:2px solid;
  }
  .errs .sub { background:rgba(239,68,68,.09);  border-color:#ef4444; color:#f7b1b1; }
  .errs .del { background:rgba(245,158,11,.09); border-color:#f59e0b; color:#f3ce93; }
  .errs .ins { background:rgba(59,130,246,.09); border-color:#3b82f6; color:#a9c8f7; }
  .sub-story {
      margin-top:.55rem; padding:.5rem .6rem;
      background:#12151c; border:1px solid var(--border); border-radius:6px;
  }
  .sub-head { font-size:.74rem; color:var(--accent); margin-bottom:.35rem; }
  .foot { font-size:.72rem; color:var(--muted); margin-top:.6rem; }
  .empty {
      border:1px dashed var(--border); border-radius:9px;
      padding:2.4rem 1.2rem; text-align:center;
      color:var(--muted); font-size:.9rem; background:#11141a;
  }
  @media (prefers-reduced-motion: reduce) {
      .entry { transition:none; }
  }
</style>

<div class="wrap">
  <div class="bar">
    <button id="tk-play" aria-label="Play">&#9654;</button>
    <input id="tk-seek" type="range" min="0" max="1000" value="0"
           step="1" aria-label="Seek">
    <span class="clock" id="tk-clock">0:00 / 0:00</span>
  </div>
  __STAGE__
  <div class="list" id="tk-list">__ENTRIES__</div>
</div>

<script>
(function () {
  var video = document.getElementById('tk-video');
  var list  = document.getElementById('tk-list');
  var play  = document.getElementById('tk-play');
  var seek  = document.getElementById('tk-seek');
  var clock = document.getElementById('tk-clock');
  var items = Array.prototype.slice.call(list.querySelectorAll('.entry'));
  var active = null;
  var scrubbing = false;

  function fmt(t) {
    if (!isFinite(t)) t = 0;
    var m = Math.floor(t / 60), s = Math.floor(t % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  // Clicking an entry jumps playback to where that text was on screen.
  items.forEach(function (el) {
    el.addEventListener('click', function () {
      if (!video) return;
      video.currentTime = parseFloat(el.dataset.start) || 0;
      video.play();
    });
  });

  if (!video) return;

  function highlight(t) {
    // Ticker stories overlap on screen, so several windows can contain t.
    // The most recently started one is the one being read out — take it.
    var best = null;
    for (var i = 0; i < items.length; i++) {
      var s = parseFloat(items[i].dataset.start);
      var e = parseFloat(items[i].dataset.end);
      if (t >= s && t < e && (best === null ||
          s >= parseFloat(best.dataset.start))) {
        best = items[i];
      }
    }
    if (best === active) return;
    if (active) active.classList.remove('tk-active');
    active = best;
    if (active) {
      active.classList.add('tk-active');
      active.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  video.addEventListener('timeupdate', function () {
    var d = video.duration || 0;
    if (!scrubbing && d) seek.value = String((video.currentTime / d) * 1000);
    clock.textContent = fmt(video.currentTime) + ' / ' + fmt(d);
    highlight(video.currentTime);
  });

  video.addEventListener('loadedmetadata', function () {
    clock.textContent = fmt(0) + ' / ' + fmt(video.duration || 0);
  });

  play.addEventListener('click', function () {
    if (video.paused) { video.play(); } else { video.pause(); }
  });
  video.addEventListener('play',  function () {
    play.innerHTML = '&#10073;&#10073;';
    play.setAttribute('aria-label', 'Pause');
  });
  video.addEventListener('pause', function () {
    play.innerHTML = '&#9654;';
    play.setAttribute('aria-label', 'Play');
  });

  seek.addEventListener('input', function () {
    scrubbing = true;
    var d = video.duration || 0;
    if (d) {
      var t = (parseFloat(seek.value) / 1000) * d;
      clock.textContent = fmt(t) + ' / ' + fmt(d);
    }
  });
  seek.addEventListener('change', function () {
    var d = video.duration || 0;
    if (d) video.currentTime = (parseFloat(seek.value) / 1000) * d;
    scrubbing = false;
  });
})();
</script>
"""


def build_sync_panel(entries, data_uri, empty_message, size_note=None):
    """One self-contained HTML block: video + transcript, wired with plain JS.

    Everything lives in the same DOM so highlighting follows playback without
    a round trip to Python — Streamlit reruns are far too coarse for this.
    """
    if data_uri:
        stage = (f'<div class="stage">'
                 f'<video id="tk-video" src="{data_uri}" preload="metadata">'
                 f'</video></div>')
        video_h = 340
    else:
        note = size_note or 'No video loaded.'
        stage = f'<div class="novideo">{html.escape(note)}</div>'
        video_h = 0

    if entries:
        blocks = []
        for e in entries:
            blocks.append(
                f'<div class="entry" data-start="{e["start"]:.3f}" '
                f'data-end="{e["end"]:.3f}">'
                f'<div class="head"><span class="name">'
                f'{html.escape(e["label"])}</span>'
                f'<span class="stamp">{html.escape(e["time"])}</span></div>'
                f'{e["body"]}</div>')
        entries_html = ''.join(blocks)
        list_h = 420
    else:
        entries_html = f'<div class="empty">{html.escape(empty_message)}</div>'
        list_h = 220

    total = video_h + list_h + 110
    return (SYNC_PANEL_TEMPLATE
            .replace('__HEIGHT__', str(total - 24))
            .replace('__VIDEOH__', str(video_h or 1))
            .replace('__STAGE__', stage)
            .replace('__ENTRIES__', entries_html)), total


def render_sync_panel(entries, empty_message):
    """Resolve the current clip and render the synced panel component."""
    video_path = st.session_state.video_path
    data_uri   = None
    size_note  = None

    if video_path and os.path.exists(video_path):
        size = os.path.getsize(video_path)
        data_uri = video_data_uri(video_path, size, os.path.getmtime(video_path))
        if data_uri is None:
            size_note = (f'Clip is {size / 1024 / 1024:.0f} MB — too large to '
                         f'embed in the player. The transcript below still '
                         f'works; playback sync needs a smaller clip.')
    else:
        size_note = 'Upload a video to play it alongside the transcript.'

    markup, height = build_sync_panel(entries, data_uri, empty_message,
                                      size_note)
    components.html(markup, height=height, scrolling=False)


def init_state():
    defaults = {
        'results'     : None,
        'result_kind' : None,
        'video_path'  : None,
        'video_name'  : None,
        'video_key'   : None,
        'script_path' : None,
        'script_key'  : None,
        'preview'     : None,
        'error'       : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def save_upload(uploaded, suffix=None):
    suffix = suffix or ('.' + uploaded.name.split('.')[-1])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def render_left(overlay_slot):
    st.markdown('<div class="tk-panel-label">Start new extraction</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload video",
                                type=["mp4", "webm", "avi", "mov"])
    if uploaded is not None:
        key = f"{uploaded.name}:{uploaded.size}"
        if key != st.session_state.video_key:
            old = st.session_state.video_path
            if old and os.path.exists(old):
                try:
                    os.unlink(old)
                except OSError:
                    pass
            st.session_state.video_path = save_upload(uploaded)
            st.session_state.video_name = uploaded.name
            st.session_state.video_key   = key
            st.session_state.preview     = None
            st.session_state.results     = None
            st.session_state.error       = None

    video_path = st.session_state.video_path
    if not video_path:
        st.caption("Upload a broadcast clip to begin. "
                   "MP4, WEBM, AVI and MOV are supported.")
        return None, None

    try:
        fw, fh, fps, duration = probe_video(video_path)
    except RuntimeError as e:
        st.error(str(e))
        return None, None

    st.caption(f"{st.session_state.video_name} — {fw}×{fh}, "
               f"{fps:.1f} fps, {seconds_to_hms(duration)}")

    segment_mode = st.toggle(
        "Segment mode", value=False,
        help="On: split the video on scene changes and read each segment. "
             "Off: read continuously from one timestamp to the end.")

    st.markdown("**Ticker region (percentages)**")
    c1, c2 = st.columns(2)
    left   = c1.number_input("left",   0.0, 1.0, 0.0,  step=0.001, format="%.3f")
    top    = c2.number_input("top",    0.0, 1.0, 0.90, step=0.001, format="%.3f")
    c3, c4 = st.columns(2)
    width  = c3.number_input("width",  0.0, 1.0, 1.0,  step=0.001, format="%.3f")
    height = c4.number_input("height", 0.0, 1.0, 0.10, step=0.001, format="%.3f")
    st.caption("left top width height — run `python get_coordinates.py "
               "<video>` to find exact values for your channel.")

    bounds_ok = True
    if width <= 0 or height <= 0:
        st.error("Width and height must be greater than 0.")
        bounds_ok = False
    elif left + width > 1.0 or top + height > 1.0:
        st.error("The region runs off the frame — "
                 "left + width and top + height must each stay within 1.0.")
        bounds_ok = False

    ticker_window = t_height = t_width = None
    if bounds_ok:
        ticker_window, t_height, t_width = percentages_to_window(
            left, top, width, height, fw, fh)

    if st.button("Preview ticker region", disabled=not bounds_ok,
                 **FULL_WIDTH):
        img = crop_preview(video_path, ticker_window)
        if img is None:
            st.session_state.preview = None
            st.warning("Could not read a frame at that region.")
        else:
            st.session_state.preview = img

    if st.session_state.preview is not None:
        st.image(st.session_state.preview,
                 caption=f"Ticker crop — {t_width}×{t_height} px")

    # The player moved to the right panel, where it shares a DOM with the
    # transcript so highlighting can follow playback (Feature 2).

    timestamp = "00:00:00"
    ts_ok = True
    if not segment_mode:
        timestamp = st.text_input("Ticker start time (HH:MM:SS)",
                                  value="00:00:00")
        if not re.match(r'^\d{2}:\d{2}:\d{2}$', timestamp.strip()):
            st.error("Use HH:MM:SS — for example 00:02:35.")
            ts_ok = False
        elif parse_timestamp(timestamp) >= duration:
            st.error(f"That is past the end of the video "
                     f"({seconds_to_hms(duration)}).")
            ts_ok = False

    script_upload = st.file_uploader("Ground truth script (optional)",
                                     type=["txt", "srt"], key="script")
    if script_upload is not None:
        key = f"{script_upload.name}:{script_upload.size}"
        if key != st.session_state.script_key:
            st.session_state.script_path = save_upload(script_upload,
                                                       suffix='.txt')
            st.session_state.script_key = key
    elif st.session_state.script_key is not None:
        st.session_state.script_path = None
        st.session_state.script_key  = None

    if st.session_state.script_path:
        st.caption("Script loaded — CER and WER will be reported.")
    else:
        st.caption("No script — extracted text only, no evaluation.")

    run = st.button("Extract ticker", type="primary",
                    disabled=not (bounds_ok and ts_ok), **FULL_WIDTH)

    if run:
        params = dict(BASE_PARAMETERS)
        st.session_state.error = None
        try:
            if segment_mode:
                # The overlay is markup only — it must be flushed to the
                # browser before the pipeline call blocks the script.
                show_extraction_overlay("Splitting on scene changes, "
                                        "then reading each segment",
                                        overlay_slot)
                res = run_segment_extraction(
                    video_path, ticker_window, t_height, t_width,
                    st.session_state.script_path, params)
                st.session_state.results = res
                st.session_state.result_kind = 'segment'
            else:
                show_extraction_overlay(f"Reading from {timestamp} "
                                        f"to the end of the clip",
                                        overlay_slot)
                res = run_full_extraction(
                    video_path, timestamp, ticker_window,
                    t_height, t_width,
                    st.session_state.script_path, params)
                st.session_state.results = res
                st.session_state.result_kind = 'full'
        except ZeroDivisionError:
            st.session_state.results = None
            st.session_state.error = (
                "No scroll speed could be estimated. The pipeline needs to see "
                "the same word move across several sampled frames — check the "
                "ticker region with Preview, and make sure the clip is long "
                "enough and the ticker is actually scrolling.")
        except IndexError:
            st.session_state.results = None
            st.session_state.error = (
                "No frames were read from that starting point. "
                "Try an earlier timestamp.")
        except Exception as e:
            st.session_state.results = None
            st.session_state.error = f"Extraction failed: {e}"
        st.rerun()

    return ticker_window, segment_mode


def render_full_results(results):
    stories = results['stories']
    ev = results.get('eval_results')

    c1, c2, c3 = st.columns(3)
    c1.metric("Stories extracted", len(stories))
    if ev and ev['full']['cer'] is not None:
        c2.metric("Overall CER", ev['full']['cer'])
        c3.metric("Overall WER", ev['full']['wer'])

    if not stories:
        st.info("No stories passed the validity filter. "
                "Try widening the ticker region or a different start time.")
        render_sync_panel([], "No stories to follow along with yet.")
        return

    st.caption("Play the clip — the story on screen right now stays "
               "highlighted. Click any story to jump to it.")
    render_sync_panel(full_entries(results), "No stories extracted.")


def render_segment_results(results):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total segments", results['total_segments'])
    c2.metric("Valid segments", results['valid_segments'])
    c3.metric("Avg fragment CER",
              results['avg_frag_cer'] if results['avg_frag_cer'] is not None
              else "—")

    if not results['segments']:
        st.info("No segment produced usable ticker text. "
                "Check the ticker region, or try full length mode.")
        render_sync_panel([], "No segments to follow along with yet.")
        return

    st.caption("Play the clip — the segment currently on screen stays "
               "highlighted. Click any segment to jump to it.")
    render_sync_panel(segment_entries(results), "No segments extracted.")


def render_right():
    st.markdown('<div class="tk-panel-label">Extracted ticker</div>',
                unsafe_allow_html=True)

    results = st.session_state.results
    kind    = st.session_state.result_kind

    c1, c2 = st.columns([1, 1])
    if c1.button("Reset", **FULL_WIDTH):
        for key in ['results', 'result_kind', 'preview', 'error']:
            st.session_state[key] = None
        st.rerun()

    if results:
        video_id = results.get('video_id', 'results')
        csv_text = (full_results_to_csv(results) if kind == 'full'
                    else segment_results_to_csv(results))
        c2.download_button("Download CSV", csv_text,
                           file_name=f"{video_id}_{kind}_results.csv",
                           mime="text/csv", **FULL_WIDTH)

    if st.session_state.error:
        st.error(st.session_state.error)
        return

    if not results:
        render_sync_panel(
            [], "Nothing extracted yet. Set the ticker region on the left, "
                "then run Extract ticker.")
        return

    if kind == 'full':
        render_full_results(results)
    else:
        render_segment_results(results)


def main():
    st.set_page_config(layout="wide",
                       page_title="Rolling Ticker Extraction",
                       page_icon="📰")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    st.markdown(
        '<div class="tk-header"><div class="tk-mark">📰</div>'
        '<div class="tk-title">Rolling Ticker Extraction</div></div>',
        unsafe_allow_html=True)

    # Root-level slot the extraction overlay renders into — see
    # show_extraction_overlay for why it cannot live inside a column.
    overlay_slot = st.empty()

    left, right = st.columns([2, 3], gap="large")
    with left:
        render_left(overlay_slot)
    with right:
        render_right()


if __name__ == "__main__":
    main()