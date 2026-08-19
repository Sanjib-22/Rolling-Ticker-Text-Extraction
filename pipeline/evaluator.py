"""
Compares extracted ticker text against ground truth script using jiwer.
Supports two evaluation modes:
  - Full: all extracted stories including boundary fragments
  - Filtered: only well-matched stories (similarity >= threshold)
"""
import re
import os
import difflib
from matplotlib.pylab import rint
from rapidfuzz import fuzz
from jiwer import cer, wer

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
    'as', 'it', 'its', 'be', 'been', 'has', 'have', 'had', 'that',
    'this', 'these', 'those', 'not', 'no', 'so', 'if', 'than',
    'then', 'when', 'where', 'who', 'which', 'what', 'how'
}

def parse_script(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    return [l for l in lines if l]

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def match_story_to_script(story: str, script_lines: list[str],
                           top_n: int = 3) -> list[tuple[str, float]]:
    """Returns best match plus genuinely related alternatives"""

    if not script_lines:
        return [('', 0.0)]

    story_clean = clean_text(story)
    story_content = set(story_clean.split()) - STOPWORDS

    scores = []
    for line in script_lines:
        line_clean = clean_text(line)
        line_content = set(line_clean.split()) - STOPWORDS

        union = story_content | line_content
        jaccard = len(story_content & line_content) / len(union) if union else 0.0
        
        seq     = fuzz.ratio(story_clean, line_clean) / 100.0
        partial = fuzz.partial_ratio(story_clean, line_clean) / 100.0
        combined = (jaccard * 0.4 + seq * 0.4 + partial * 0.2) * 100
        scores.append((line, round(combined, 2)))

    scores.sort(key=lambda x: x[1], reverse=True)

    # Always include best match; add alternatives only if genuinely related
    result = [scores[0]]
    for line, score in scores[1:top_n]:
        if score >= 50:
            result.append((line, score))

    return result


def get_word_errors(extracted: str, reference: str) -> list[dict]:
    """Task 2 — Word-level diff between extracted and reference."""
    ext_words = clean_text(extracted).split()
    ref_words = clean_text(reference).split()

    matcher = difflib.SequenceMatcher(None, ref_words, ext_words)
    errors = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            errors.append({
                'type': 'SUB',
                'expected': ' '.join(ref_words[i1:i2]),
                'got': ' '.join(ext_words[j1:j2])
            })
        elif tag == 'delete':
            errors.append({
                'type': 'DEL',
                'expected': ' '.join(ref_words[i1:i2]),
                'got': None
            })
        elif tag == 'insert':
            errors.append({
                'type': 'INS',
                'expected': None,
                'got': ' '.join(ext_words[j1:j2])
            })
    return errors

def evaluate_story(extracted: str, reference: str) -> dict:
    ext_clean = clean_text(extracted)
    ref_clean = clean_text(reference)

    if not ref_clean:
        return {
            'cer': 1.0, 'wer': 1.0,
            'substitutions': 0, 'insertions': 0, 'deletions': 0,
            'word_errors': []
        }

    try:
        from jiwer import process_words
        cer_score = round(cer(ref_clean, ext_clean), 4)
        wer_score = round(wer(ref_clean, ext_clean), 4)
        output    = process_words(ref_clean, ext_clean)
        subs = output.substitutions
        ins  = output.insertions
        dels = output.deletions
    except Exception:
        cer_score = 1.0
        wer_score = 1.0
        subs = ins = dels = 0

    word_errors = get_word_errors(extracted, reference)

    return {
        'cer': cer_score,
        'wer': wer_score,
        'substitutions': subs,
        'insertions': ins,
        'deletions': dels,
        'word_errors': word_errors
    }

def evaluate_fragment(extracted: str, script_lines: list[str]) -> dict:
    ext_clean  = clean_text(extracted)
    ext_words  = ext_clean.split()
    ext_len    = len(ext_words)

    if not ext_words or not script_lines:
        return {
            'cer': 1.0, 'wer': 1.0,
            'aligned_reference': '',
            'matched_line': '',
            'similarity': 0.0
        }

    best_cer        = 1.0
    best_wer        = 1.0
    best_window     = ''
    best_line       = ''
    best_similarity = 0.0

    for line in script_lines:
        ref_clean = clean_text(line)
        ref_words = ref_clean.split()

        # If extracted is longer than reference — compare directly
        if ext_len >= len(ref_words):
            c = round(cer(ref_clean, ext_clean), 4)
            w = round(wer(ref_clean, ext_clean), 4)
            sim = fuzz.token_set_ratio(ext_clean, ref_clean)
            if c < best_cer:
                best_cer        = c
                best_wer        = w
                best_window     = ref_clean
                best_line       = line
                best_similarity = float(sim)
            continue

        # Slide window of ext_len words across reference
        for i in range(len(ref_words) - ext_len + 1):
            window      = ' '.join(ref_words[i: i + ext_len])
            window_cer  = cer(window, ext_clean)
            if window_cer < best_cer:
                best_cer        = round(window_cer, 4)
                best_wer        = round(wer(window, ext_clean), 4)
                best_window     = window
                best_line       = line
                best_similarity = float(
                    fuzz.ratio(ext_clean, window)
                )

    return {
        'cer'               : best_cer,
        'wer'               : best_wer,
        'aligned_reference' : best_window,
        'matched_line'      : best_line,
        'similarity'        : round(best_similarity, 2)
    }


def evaluate_segments(output_text: str, script_path: str) -> dict:
    """
    Segment-aware evaluation using fragment alignment.
    For each extracted story, finds the best-aligned window within
    any script line rather than comparing against the full line.
    """
    script_lines = parse_script(script_path)

    if not script_lines:
        return {
            'per_story'    : [],
            'fragment_cer' : None,
            'fragment_wer' : None,
            'count'        : 0
        }

    stories = [s.strip() for s in output_text.split('|') if s.strip()]

    if not stories:
        return {
            'per_story'    : [],
            'fragment_cer' : None,
            'fragment_wer' : None,
            'count'        : 0
        }

    per_story = []
    for i, story in enumerate(stories):
        frag = evaluate_fragment(story, script_lines)
        word_errors = get_word_errors(story, frag['aligned_reference'])

        per_story.append({
            'story_index'       : i + 1,
            'extracted'         : story,
            'aligned_reference' : frag['aligned_reference'],
            'matched_line'      : frag['matched_line'],
            'similarity'        : frag['similarity'],
            'fragment_cer'      : frag['cer'],
            'fragment_wer'      : frag['wer'],
            'word_errors'       : word_errors
        })

    valid = [s for s in per_story if s['fragment_cer'] is not None]
    avg_cer = round(
        sum(s['fragment_cer'] for s in valid) / len(valid), 4
    ) if valid else None
    avg_wer = round(
        sum(s['fragment_wer'] for s in valid) / len(valid), 4
    ) if valid else None

    return {
        'per_story'    : per_story,
        'fragment_cer' : avg_cer,
        'fragment_wer' : avg_wer,
        'count'        : len(per_story)
    }

def evaluate_all(output_text: str, script_path: str,
                 boundary_threshold: float = 50.0,
                 frequencies: list | None = None) -> dict:
    """
    Main evaluation function.
    Splits output_text by | into stories, matches each to script, computes CER/WER per story.
    """
    script_lines = parse_script(script_path)

    if not script_lines:
        print(f"  [Evaluator] No script found at {script_path} — skipping")
        return {
            'per_story': [],
            'full': {'cer': None, 'wer': None, 'count': 0},
            'filtered': {'cer': None, 'wer': None, 'count': 0}
        }

    stories = [s.strip() for s in output_text.split('|') if s.strip()]

    if not stories:
        return {
            'per_story': [],
            'full': {'cer': None, 'wer': None, 'count': 0},
            'filtered': {'cer': None, 'wer': None, 'count': 0}
        }

    per_story = []
    for i, story in enumerate(stories):
        # Task 1 — top-N matches
        top_matches = match_story_to_script(story, script_lines, top_n=3)
        best_match, best_similarity = top_matches[0]
        alt_matches = top_matches[1:]

        scores = evaluate_story(story, best_match)
        is_boundary = best_similarity < boundary_threshold

        story_freq = None
        if frequencies is not None and i < len(frequencies):
            story_freq = frequencies[i]

        per_story.append({
            'story_index': i + 1,
            'extracted': story.strip(),
            'matched_reference': best_match,
            'similarity': round(best_similarity, 2),
            'alt_matches': [(line, round(score, 2)) for line, score in alt_matches],
            'cer': scores['cer'],
            'wer': scores['wer'],
            'substitutions': scores['substitutions'],
            'insertions': scores['insertions'],
            'deletions': scores['deletions'],
            'word_errors': scores['word_errors'],
            'is_boundary': is_boundary
        })

    # Full evaluation — all stories
    full_cer = round(sum(s['cer'] for s in per_story) / len(per_story), 4)
    full_wer = round(sum(s['wer'] for s in per_story) / len(per_story), 4)

    # Filtered evaluation — exclude boundary fragments
    filtered = [s for s in per_story if not s['is_boundary']]
    if filtered:
        filt_cer = round(sum(s['cer'] for s in filtered) / len(filtered), 4)
        filt_wer = round(sum(s['wer'] for s in filtered) / len(filtered), 4)
    else:
        filt_cer = None
        filt_wer = None

    return {
        'per_story': per_story,
        'full': {
            'cer': full_cer,
            'wer': full_wer,
            'count': len(per_story)
        },
        'filtered': {
            'cer': filt_cer,
            'wer': filt_wer,
            'count': len(filtered),
            'excluded': len(per_story) - len(filtered)
        }
    }

def print_results(results: dict, video_id: str, frequencies: list | None = None):
    if not results['per_story']:
        print("  [Evaluator] No results to display")
        return

    print(f"\n{'─' * 64}")
    print(f"EVALUATION RESULTS — {video_id}")
    print(f"{'─' * 64}")

    for i, s in enumerate(results['per_story']):
        boundary_tag = " [boundary fragment]" if s['is_boundary'] else ""
        print(f"\n  Story {s['story_index']}{boundary_tag}:")
        print(f"    Extracted : {s['extracted']}")
        print(f"    Reference : {s['matched_reference']}")
        print(f"    Similarity: {s['similarity']}%")
        if frequencies and i < len(frequencies) and frequencies[i] is not None:
            print(f"    Frequency : visible across ~{frequencies[i]} frames")

        # Alternative matches
        if s.get('alt_matches') and s['similarity'] <= 65:
            print(f"    Alt matches:")
            for line, score in s['alt_matches']:
                print(f"      ({score}%) {line}")

        # Error summary
        print(f"    -> Error Summary: {s.get('substitutions', 0)} substitutions, "
              f"{s.get('deletions', 0)} deletions, "
              f"{s.get('insertions', 0)} insertions")

        # Detailed word errors
        word_errors = s.get('word_errors', [])
        if word_errors:
            print(f"    -> Detailed Word Errors:")
            for err in word_errors:
                if err['type'] == 'SUB':
                    print(f"       [SUB] expected '{err['expected']}' "
                          f"but got '{err['got']}'")
                elif err['type'] == 'DEL':
                    print(f"       [DEL] expected '{err['expected']}' "
                          f"but got nothing")
                elif err['type'] == 'INS':
                    print(f"       [INS] hallucinated '{err['got']}'")
        else:
            print(f"    -> Perfect extraction")

        print(f"    CER: {s['cer']}  |  WER: {s['wer']}")

    print(f"\n  {'─' * 40}")
    print(f"  Full evaluation    ({results['full']['count']} stories):")
    print(f"    CER: {results['full']['cer']}  |  WER: {results['full']['wer']}")

    if results['filtered']['cer'] is not None:
        excl = results['filtered']['excluded']
        print(f"\n  Filtered evaluation ({results['filtered']['count']} stories,"
              f" {excl} boundary fragment{'s' if excl != 1 else ''} excluded):")
        print(f"    CER: {results['filtered']['cer']}  |  "
              f"WER: {results['filtered']['wer']}")
    else:
        print(f"\n  Filtered evaluation: no well-matched stories found")

    print(f"{'─' * 64}\n")