"""
Reads master files and generates broadcast_summary.txt.
Run from inside the code/ folder: python broadcast_summarizer.py
"""
import csv
import os
from datetime import datetime

MASTER_DIR = './samples/master_files'
OUTPUT_PATH = f'{MASTER_DIR}/broadcast_summary.txt'

def read_csv(path):
  if not os.path.exists(path):
    return []
  with open(path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    return list(reader)

def generate_summary():
  output_rows = read_csv(f'{MASTER_DIR}/master_output.csv')
  meta_rows   = read_csv(f'{MASTER_DIR}/master_metadata.csv')
  eval_rows   = read_csv(f'{MASTER_DIR}/master_evaluation.csv')

  if not output_rows:
    print("No data found — run easyrun.py on at least one sample first.")
    return

  meta_lookup = {}
  for row in meta_rows:
    key = (row['video_id'], row['seek_time'])
    meta_lookup[key] = row.get('run_timestamp', 'unknown')

  eval_lookup = {}
  for row in eval_rows:
    key = (row['video_id'], row['seek_time'])
    eval_lookup.setdefault(key, []).append(row)

  grouped = {}
  for row in output_rows:
    vid = row['video_id']
    grouped.setdefault(vid, []).append(row)

  sorted_video_ids = sorted(grouped.keys())

  lines = []
  lines.append('=' * 64)
  lines.append('BROADCAST SUMMARY')
  lines.append(f'Generated  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
  lines.append(f'Samples    : {len(sorted_video_ids)}')
  lines.append(f'Total runs : {len(output_rows)}')
  lines.append('=' * 64)

  for vid in sorted_video_ids:
    lines.append('')
    lines.append(f'[{vid}]')
    lines.append('─' * 48)

    for row in grouped[vid]:
      seek    = row['seek_time']
      text    = row['extracted_text'].strip()
      ts      = meta_lookup.get((vid, seek), 'unknown')
      full_cer = row.get('full_cer', 'N/A')
      full_wer = row.get('full_wer', 'N/A')
      filt_cer = row.get('filtered_cer', 'N/A')
      filt_wer = row.get('filtered_wer', 'N/A')

      lines.append(f'  Seek     : {seek}')
      lines.append(f'  Run at   : {ts}')
      lines.append(f'  Full eval    — CER: {full_cer}  WER: {full_wer}')
      lines.append(f'  Filtered eval — CER: {filt_cer}  WER: {filt_wer}')
      lines.append(f'  Content  :')

      stories = [s.strip() for s in text.split('|') if s.strip()]
      for story in stories:
        lines.append(f'    {story}')

      # Per story evaluation
      eval_data = eval_lookup.get((vid, seek), [])
      if eval_data:
        lines.append(f'  Per-story evaluation:')
        for s in eval_data:
          boundary = ' [boundary]' if s.get('is_boundary', '').lower() == 'true' else ''
          lines.append(f'    Story {s["story_index"]}{boundary}:')
          lines.append(f'      Extracted : {s["extracted"]}')
          lines.append(f'      Reference : {s["matched_reference"]}')
          lines.append(f'      CER: {s["cer"]}  WER: {s["wer"]}  Sim: {s["similarity"]}%')
      lines.append('')

  lines.append('=' * 64)
  lines.append('END OF BROADCAST SUMMARY')
  lines.append('=' * 64)

  summary = '\n'.join(lines)

  with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(summary)

  print(summary)
  print(f'\nSummary written to {OUTPUT_PATH}')

if __name__ == '__main__':
  generate_summary()