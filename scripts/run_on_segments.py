"""
run_on_segments.py
Runs video segmentor first, then ticker extraction on each segment.
Uses frame_count override to limit extraction to segment duration only.
Dynamic jump_size adjusts based on segment length for reliable speed estimation.

Usage:
  python run_on_segments.py samples/sample_3.mp4
  python run_on_segments.py samples/sample_4.mp4

Update COORDS per channel:
  DD India    : (0.189, 0.908, 0.766, 0.053)
  CNN News18  : (0.00,  0.908, 0.998, 0.046)
"""
import sys
import os
import csv
import re
from datetime import datetime
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pipeline'))

from video_segmentor import VideoSegmentation
from slidingreader import read_ticker, add_story_start_end_times
from helpers import concatenate_news_stories
from video import Video
from tesseract import TesseractOCR
from evaluator import evaluate_all, evaluate_segments, print_results

# ── Config — update per sample ────────────────────────────────────────
VIDEO_PATH    = sys.argv[1] if len(sys.argv) > 1 else 'samples/sample_6.mp4'
COORDS        = (0.032, 0.922, 0.802, 0.053)  # DD India sample_3

SEG_METHOD    = 'Bhattachrya Distance'
SEG_THRESHOLD = 0.7
MIN_SEG_SECS  = 3.0    # skip segments shorter than this
TARGET_SAMPLES = 80    # minimum sampled frames per segment for speed estimation
MASTER_DIR    = './samples/master_files'
os.makedirs(MASTER_DIR, exist_ok=True)

video_id    = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
script_path = f'samples/{video_id}_script.txt'

base_parameters = {
    'resize_font'     : True,
    'height'          : None,
    'new_height'      : 32,
    'interpolation'   : 'cubic',
    'add_padding'     : False,
    'method'          : None,
    'gamma_correct'   : True,
    'merge_method'    : 'confidence',
    'garbage_method'  : 'char_confidence',
    'garbage_threshold': 0.80,
    'overlap_method'  : 'merges',
    'video_id'        : video_id,
    'master_tsv_path' : f'{MASTER_DIR}/master_words.tsv'
}

# ── Helpers ───────────────────────────────────────────────────────────
def seconds_to_hms(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def clean_story(text):
    words = text.split()
    if not words:
        return text
    result = [words[0]]
    for word in words[1:]:
        if word.lower() != result[-1].lower():
            result.append(word)
    return ' '.join(result).strip()

def is_valid_story(text):
    words = text.split()
    if len(words) < 3:
        return False
    if len([w for w in words if len(w) >= 3]) < 2:
        return False
    if len(text) < 15:
        return False
    return True

def clean_output(text):
    text = re.sub(r'[\{\[\(]', '', text)
    text = re.sub(r'\|+', '|', text)
    text = re.sub(r'\s*\|\s*', ' | ', text)
    text = re.sub(r'^\s*\|\s*', '', text)
    text = re.sub(r'\s*\|\s*$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Step 1: Run video segmentor ───────────────────────────────────────
print(f"\n{'='*60}")
print(f"VIDEO SEGMENTATION — {video_id}")
print(f"{'='*60}")

seg = VideoSegmentation(
    path=VIDEO_PATH,
    no_of_bins=16,
    frame_skip=28,
    threshold=SEG_THRESHOLD
)
distances, fps, skip, duration, segment_frames, segment_time = seg.segment_video(
    method=SEG_METHOD
)

print(f"Duration       : {duration:.2f}s")
print(f"Scene changes  : {len(segment_time)}")
print(f"Timestamps     : {[round(t,2) for t in segment_time]}")

# Build segment boundaries
boundaries = [0.0] + segment_time + [duration]
segments   = [(boundaries[i], boundaries[i+1])
              for i in range(len(boundaries)-1)]
print(f"Total segments : {len(segments)}")

# ── Step 2: Get ticker window ─────────────────────────────────────────
left_pct, top_pct, width_pct, height_pct = COORDS
probe = Video(VIDEO_PATH)
fh, fw = probe.height, probe.width
try:
    probe.video_capture.release()
except:
    pass
del probe

minx = int(left_pct * fw)
miny = int(top_pct  * fh)
maxx = int((left_pct + width_pct)  * fw) - 1
maxy = int((top_pct  + height_pct) * fh) - 1
ticker_window = [[miny, maxy], [minx, maxx]]
t_height = maxy - miny + 1
t_width  = maxx - minx + 1

print(f"Ticker window  : x={minx}-{maxx}, y={miny}-{maxy}\n")

# ── Step 3: Extract ticker per segment ───────────────────────────────
print(f"{'='*60}")
print(f"TICKER EXTRACTION PER SEGMENT")
print(f"{'='*60}")

results_per_segment = []

for i, (start, end) in enumerate(segments):
    seg_dur = end - start
    seg_label = f"Segment {i+1} [{seconds_to_hms(start)} → {seconds_to_hms(end)}] ({seg_dur:.1f}s)"

    if seg_dur < MIN_SEG_SECS:
        print(f"\n{seg_label} — skipped (too short)")
        continue

    print(f"\n{seg_label}")

    try:
        # ── Dynamic jump_size ─────────────────────────────────────────
        seg_frame_count = int(seg_dur * fps)
        dynamic_jump    = max(1, seg_frame_count // TARGET_SAMPLES)
        print(f"  frames={seg_frame_count}  jump_size={dynamic_jump}  "
              f"sampled≈{seg_frame_count // dynamic_jump}")

        # ── Build parameters for this segment ────────────────────────
        params = dict(base_parameters)
        params['jump_size'] = dynamic_jump
        params['height']    = t_height
        params['width']     = t_width

        # ── Load video, seek, override frame_count ───────────────────
        video = Video(VIDEO_PATH)
        video.seek(start)
        video.frame_count = seg_frame_count
        seek_frame = int(video.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        video._seek_offset = seek_frame  # store offset
        original_frame = video.frame.__func__

        def offset_frame(self, frame_number=None):
            if frame_number is not None:
                frame_number = frame_number + self._seek_offset
            return original_frame(self, frame_number)

        import types
        video.frame = types.MethodType(offset_frame, video)

        ocr = TesseractOCR(**params)
        ocr._preprocesses['height'] = t_height

        # ── Extract ──────────────────────────────────────────────────
        raw_stories = read_ticker(video, ticker_window, ocr, **params)
        add_story_start_end_times(raw_stories, video)
        try:
            video.video_capture.release()
        except:
            pass
        del video

        cleaned = []
        for story in raw_stories:
            story['text'] = clean_story(story['text'])
            cleaned.append(story)

        stories = [s for s in cleaned if is_valid_story(s['text'])]

        if not stories:
            print(f"  No valid stories extracted")
            continue

        raw_output = concatenate_news_stories(stories, char=' | ')
        output     = clean_output(raw_output)
        print(f"  Extracted: {output[:120]}{'...' if len(output) > 120 else ''}")

        # ── Evaluate — fragment-aware ─────────────────────────────────
        seg_eval    = evaluate_segments(output, script_path)
        frag_cer    = seg_eval['fragment_cer']
        frag_wer    = seg_eval['fragment_wer']

        if frag_cer is not None:
            print(f"  Fragment CER: {frag_cer}  WER: {frag_wer} "
                  f"({seg_eval['count']} stories, window-aligned)")

        # ── Per-story detail ──────────────────────────────────────────
        for s in seg_eval['per_story']:
            print(f"    Story {s['story_index']}: "
                  f"sim={s['similarity']}%  "
                  f"CER={s['fragment_cer']}  WER={s['fragment_wer']}")
            print(f"      Extracted : {s['extracted']}")
            print(f"      Aligned   : {s['aligned_reference']}")
            if s.get('word_errors'):
                for err in s['word_errors'][:3]:
                    if err['type'] == 'SUB':
                        print(f"      [SUB] '{err['expected']}' → '{err['got']}'")
                    elif err['type'] == 'DEL':
                        print(f"      [DEL] '{err['expected']}' missing")
                    elif err['type'] == 'INS':
                        print(f"      [INS] '{err['got']}' hallucinated")

        results_per_segment.append({
            'segment'     : i + 1,
            'start'       : seconds_to_hms(start),
            'end'         : seconds_to_hms(end),
            'duration'    : round(seg_dur, 2),
            'jump_size'   : dynamic_jump,
            'stories'     : len(stories),
            'extracted'   : output,
            'fragment_cer': frag_cer if frag_cer is not None else 'N/A',
            'fragment_wer': frag_wer if frag_wer is not None else 'N/A',
        })

    except Exception as e:
        print(f"  Error: {e}")
        continue

# ── Step 4: Save CSV ──────────────────────────────────────────────────
if results_per_segment:
    out_path = f'{MASTER_DIR}/{video_id}_segment_results.csv'
    fieldnames = ['segment','start','end','duration','jump_size',
                  'stories','extracted',
                  'fragment_cer','fragment_wer']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_per_segment)
    print(f"\nResults saved → {out_path}")

# ── Step 5: Print summary table ───────────────────────────────────────
print(f"\n{'='*60}")
print(f"SEGMENT SUMMARY — {video_id}")
print(f"{'='*60}")
print(f"{'Seg':<5} {'Start':<10} {'End':<10} {'Dur':>6} "
      f"{'Jump':>5} {'Stories':>7} {'Frag CER':>8} {'Frag WER':>8}")
print(f"{'─'*60}")
for r in results_per_segment:
    print(f"{r['segment']:<5} {r['start']:<10} {r['end']:<10} "
          f"{r['duration']:>6} {r['jump_size']:>5} {r['stories']:>7} "
          f"{str(r['fragment_cer']):>8} {str(r['fragment_wer']):>8}")
print(f"{'='*60}")

if results_per_segment:
    valid = [r for r in results_per_segment
             if isinstance(r['fragment_cer'], float)]
    if valid:
        avg_cer = round(sum(r['fragment_cer'] for r in valid) / len(valid), 4)
        avg_wer = round(sum(r['fragment_wer'] for r in valid) / len(valid), 4)
        print(f"Average across {len(valid)} segments: "
              f"Fragment CER={avg_cer}  WER={avg_wer}")