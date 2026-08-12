"""
helpers.py — small, generic, no-domain-specific utilities.

Merged from three former single-purpose files (storyops.py, utils.py,
tsv.py) — none of these had any real conceptual identity of their own,
just grouped one-off helper functions, so combining them costs nothing.
(Contrast with ocr.py/tesseract.py, which stayed separate on purpose —
see the comment in tesseract.py.)
"""

import os
import csv


# ── from storyops.py ────────────────────────────────────────────────
def concatenate_news_stories(l, char='\n'):
    """
    Concatenates news stories into one string
    Returns string.
    """
    result = ''
    for story in l:
        if len(result) != 0:
            result += char
        result += story['text']

    return result


# ── from utils.py ───────────────────────────────────────────────────
def makedirs(path):
  if not os.path.exists(path):
    os.makedirs(path)

def cleandir(path):
  for file in os.listdir(path):
    os.remove(path + '/' + file)

def inrange(val, minval, maxval):
    return (val >= minval) & (val <= maxval)

def requests(method, kwargs):
  if method in kwargs and kwargs[method] == True:
    return True
  else:
    return False


# ── from tsv.py ─────────────────────────────────────────────────────
"""
Helper functions for operating with TSV files
"""

def read(path):
  file = open(path, 'r', encoding = 'utf-8')
  reader = csv.reader(file, delimiter = '\t', quoting = csv.QUOTE_NONE)
  result = []
  header = reader.__next__()
  for values in reader:
    entry = {}
    for i in range(len(header)):
      entry[header[i]] = values[i]
    result.append(entry)
  file.close()
  return result

def write(l, path, columns):
  file = open(path, 'w', newline = '', encoding = 'utf-8')
  writer = csv.writer(file, delimiter = '\t', quoting = csv.QUOTE_NONE)
  row = []
  for col in columns:
    row.append(col)
  writer.writerow(row)
  for entry in l:
    row = []
    for col in columns:
      row.append(entry[col])
    writer.writerow(row)
  file.close()
