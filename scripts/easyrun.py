import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pipeline'))

from slidingreader import read_ticker, add_story_start_end_times
from helpers import concatenate_news_stories
from video import Video
from evaluator import evaluate_all, print_results
from tesseract import TesseractOCR
import numpy as np
import cv2
import csv
import os
import re
import sys
import getopt
from datetime import datetime

def cl_help():
  print('Usage: easyrun.py -o outputfile.txt inputfile.mp4')

try:
  options, args = getopt.getopt(sys.argv[1:], 'ho:')
except getopt.GetoptError:
  cl_help()
  sys.exit(2)

if len(args) != 1:
  cl_help()
  sys.exit(2)

input_name = args[0]
output_name = 'output.txt'

for option, arg in options:
  if option == '-h':
    cl_help()
    sys.exit()
  elif option == '-o':
    output_name = arg

video_id = os.path.splitext(os.path.basename(input_name))[0]
script_path = os.path.join('samples', f'{video_id}_script.txt')

MASTER_DIR = './samples/master_files'
os.makedirs(MASTER_DIR, exist_ok=True)

best_parameters = {
    'jump_size': 12,
    'resize_font': True,
    'height': None,
    'new_height': 32,
    'interpolation': 'cubic',
    'add_padding': False,
    'method': None,
    'gamma_correct': True,
    'merge_method': 'confidence',
    'garbage_method': 'char_confidence',
    'garbage_threshold': 0.80,
    'overlap_method': 'merges',
    'video_id': video_id,
    'master_tsv_path': f'{MASTER_DIR}/master_words.tsv'
}

def parse_timestamp(ts: str) -> float:
  parts = ts.strip().split(':')
  if len(parts) == 3:
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
  elif len(parts) == 2:
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

def clean_output(text: str) -> str:
  text = re.sub(r'[\{\[\(]', '', text)
  text = re.sub(r'\|+', '|', text)
  text = re.sub(r'\s*\|\s*', ' | ', text)
  text = re.sub(r'^\s*\|\s*', '', text)
  text = re.sub(r'\s*\|\s*$', '', text)
  text = re.sub(r'\s+', ' ', text).strip()
  return text

def add_story_frequency(stories, jump_size=12):
  """Add frequency field — how many sampled frames each story was visible."""
  if not jump_size or jump_size < 1:
    jump_size = 1
  for story in stories:
    min_f = story.get('minframe', 0)
    max_f = story.get('maxframe', 0)
    story['frequency'] = max(1, (max_f - min_f) // jump_size)

def get_user_ticker(frame_height: int, frame_width: int):
    print("\n  Ticker detection:")
    print("\n  Manual coordinates can be provided as percentages of the screen.")

    print("\nEnter ticker region (0.0 to 1.0):")
    print("  Format : left top width height")

    while True:
        try:
            vals = input("\n  Enter values (space separated): ").strip().split()
            if len(vals) != 4:
                print("  Need exactly 4 values — left top width height")
                continue

            left_pct   = float(vals[0])
            top_pct    = float(vals[1])
            width_pct  = float(vals[2])
            height_pct = float(vals[3])

            if not (0.0 <= left_pct < 1.0):
                print("  left must be between 0.0 and 1.0")
                continue
            if not (0.0 <= top_pct < 1.0):
                print("  top must be between 0.0 and 1.0")
                continue
            if not (0.0 < width_pct <= 1.0):
                print("  width must be between 0.0 and 1.0")
                continue
            if not (0.0 < height_pct <= 1.0):
                print("  height must be between 0.0 and 1.0")
                continue
            if left_pct + width_pct > 1.0:
                print("  left + width exceeds screen width")
                continue
            if top_pct + height_pct > 1.0:
                print("  top + height exceeds screen height")
                continue

            # Convert percentages to pixels
            minx = int(left_pct  * frame_width)
            miny = int(top_pct   * frame_height)
            maxx = int((left_pct  + width_pct)  * frame_width)  - 1
            maxy = int((top_pct   + height_pct) * frame_height) - 1

            ticker_window = [[miny, maxy], [minx, maxx]]
            t_height = maxy - miny + 1
            t_width  = maxx - minx + 1

            print(f"\n  Ticker region set:")
            print(f"    Pixels : x={minx}-{maxx}, y={miny}-{maxy}")
            print(f"    Size   : {t_width}x{t_height}px")
            confirm = input("  Confirm? (Y/n): ").strip().lower()
            if confirm == 'n':
                continue

            return ticker_window, t_height, t_width

        except (ValueError, IndexError):
            print("\nInvalid input — enter 4 numbers e.g. 0.10 0.925 0.90 0.037")

def append_to_master_output(master_dir, video_id, seek_hms, output_text,
                             full_cer, full_wer, filt_cer, filt_wer):
  path = f'{master_dir}/master_output.csv'
  file_exists = os.path.exists(path)
  with open(path, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
      writer.writerow(['video_id', 'seek_time', 'extracted_text',
                       'full_cer', 'full_wer', 'filtered_cer', 'filtered_wer'])
    writer.writerow([video_id, seek_hms, output_text,
                     full_cer if full_cer is not None else 'N/A',
                     full_wer if full_wer is not None else 'N/A',
                     filt_cer if filt_cer is not None else 'N/A',
                     filt_wer if filt_wer is not None else 'N/A'])

def append_to_master_metadata(master_dir, video_id, seek_hms):
  path = f'{master_dir}/master_metadata.csv'
  file_exists = os.path.exists(path)
  with open(path, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
      writer.writerow(['video_id', 'seek_time', 'run_timestamp'])
    writer.writerow([video_id, seek_hms,
                     datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

def append_to_master_evaluation(master_dir, video_id, seek_hms, per_story):
  path = f'{master_dir}/master_evaluation.csv'
  file_exists = os.path.exists(path)
  with open(path, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
      writer.writerow(['video_id', 'seek_time', 'story_index', 'frequency',
                       'extracted', 'matched_reference', 'similarity',
                       'alt_match_1', 'alt_score_1',
                       'alt_match_2', 'alt_score_2',
                       'cer', 'wer',
                       'substitutions', 'insertions', 'deletions',
                       'word_errors', 'is_boundary'])
    for s in per_story:
      alt1_text  = s['alt_matches'][0][0] if len(s.get('alt_matches', [])) > 0 else ''
      alt1_score = s['alt_matches'][0][1] if len(s.get('alt_matches', [])) > 0 else ''
      alt2_text  = s['alt_matches'][1][0] if len(s.get('alt_matches', [])) > 1 else ''
      alt2_score = s['alt_matches'][1][1] if len(s.get('alt_matches', [])) > 1 else ''

      error_strs = []
      for err in s.get('word_errors', []):
        if err['type'] == 'SUB':
          error_strs.append(f"[SUB] '{err['expected']}' -> '{err['got']}'")
        elif err['type'] == 'DEL':
          error_strs.append(f"[DEL] '{err['expected']}'")
        elif err['type'] == 'INS':
          error_strs.append(f"[INS] '{err['got']}'")
      word_errors_str = ' | '.join(error_strs) if error_strs else 'none'

      writer.writerow([video_id, seek_hms, s['story_index'], 
                       s.get('frequency', ''),
                       s['extracted'], s['matched_reference'],
                       s['similarity'],
                       alt1_text, alt1_score,
                       alt2_text, alt2_score,
                       s['cer'], s['wer'],
                       s.get('substitutions', 0),
                       s.get('insertions', 0),
                       s.get('deletions', 0),
                       word_errors_str, s['is_boundary']])

video = Video(input_name, language='English')
total_seconds = video.frame_count * video.frame_duration
duration_hms = seconds_to_hms(total_seconds)

print(f"\nVideo    : {input_name}")
print(f"Duration : {duration_hms}")
if os.path.exists(script_path):
  print(f"Script   : {script_path} found — evaluation will be performed")
else:
  print(f"Script   : {script_path} not found — evaluation will be skipped")
print()

while True:
  user_input = input(f"Enter ticker start time (HH:MM:SS) [max {duration_hms}]: ").strip()
  try:
    seek_seconds = parse_timestamp(user_input)
    if seek_seconds < 0 or seek_seconds >= total_seconds:
      print(f"  Out of range — enter a time between 00:00:00 and {duration_hms}")
      continue
    break
  except (ValueError, IndexError):
    print("  Invalid format — use HH:MM:SS (e.g. 00:02:35)")

seek_hms = seconds_to_hms(seek_seconds)
print(f"\nStarting extraction from {seek_hms}...\n")
video.seek(seek_seconds)

first_frame = video.frame(0)
fh, fw = first_frame.shape[:2]
user_ticker = get_user_ticker(fh, fw)

if user_ticker is not None:
    # Manual coordinates 
    ticker_window, t_height, t_width = user_ticker
    first_frame = video.frame(0)
    preview = first_frame[
        ticker_window[0][0]:ticker_window[0][1]+1,
        ticker_window[1][0]:ticker_window[1][1]+1
    ]
    # Get a preview image of the ticker region for verification of the cropping
    preview_uint8 = (preview * 255).astype(np.uint8) if preview.max() <= 1.0 else preview.astype(np.uint8)
    cv2.imwrite('samples/ticker_preview.png', preview_uint8)
    print("\nTicker preview saved to samples/ticker_preview.png")
    input("Press Enter to continue or Ctrl+C to cancel and re-enter coordinates...")
    
    ocr = TesseractOCR(**best_parameters)
    best_parameters['height'] = t_height
    best_parameters['width'] = t_width
    ocr._preprocesses['height'] = t_height
    raw_stories = read_ticker(video, ticker_window, ocr, **best_parameters)
    add_story_start_end_times(raw_stories, video)
    add_story_frequency(raw_stories, jump_size=best_parameters['jump_size'])
    stories = raw_stories

cleaned = []
for story in stories:
  story['text'] = clean_story(story['text'])
  cleaned.append(story)

def is_valid_story(text: str) -> bool:
  words = text.split()
  if len(words) < 3:
    return False
  real_words = [w for w in words if len(w) >= 3]
  if len(real_words) < 2:
    return False
  if len(text) < 15:
    return False
  return True

stories = [s for s in cleaned if is_valid_story(s['text'])]
raw_output = concatenate_news_stories(stories, char=' | ')
output = clean_output(raw_output)

# Frequencies in same order as stories appear in output
story_frequencies = [s.get('frequency') for s in stories]

with open(output_name, 'w', encoding='utf-8') as f:
  f.write(output)

print(f"Output written to {output_name}")
print(f"\nExtracted:\n{output}\n")

results = evaluate_all(output, script_path, frequencies=story_frequencies)
print_results(results, video_id, frequencies=story_frequencies)

append_to_master_output(
    MASTER_DIR, video_id, seek_hms, output,
    results['full']['cer'], results['full']['wer'],
    results['filtered']['cer'], results['filtered']['wer']
)
append_to_master_metadata(MASTER_DIR, video_id, seek_hms)

if results['per_story']:
  append_to_master_evaluation(MASTER_DIR, video_id, seek_hms,
                              results['per_story'])

print(f"Master files updated in {MASTER_DIR}/")