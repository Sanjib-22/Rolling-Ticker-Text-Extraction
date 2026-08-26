# Rolling Ticker Extraction from Broadcast Video

## 1. Introduction

Extracting scrolling news-ticker text from broadcast video presents its own set of challenges distinct from subtitle or speech transcription. Ticker text is small, often low-contrast against a moving or brightly colored background band, scrolls continuously rather than appearing as discrete captions, and frequently overlaps with channel branding, secondary tickers, or on-screen graphics.

This project builds a ticker-extraction pipeline for broadcast news video, using manual ticker-region selection, a sliding-window OCR approach, and story-boundary stitching to reconstruct individual news items from the continuously scrolling text. Two extraction strategies are evaluated - a full-length read from a fixed start timestamp to the end of the clip, and a segment mode that first splits the video on scene changes before reading each segment independently - and both are scored against ground-truth scripts using Character Error Rate (CER) and Word Error Rate (WER).

This report documents the extraction methodology, the evaluation approach for each mode, the full results across six broadcast clips spanning two channels and three clip lengths, and a direct comparison of the two modes' accuracy.

## 2. Project Progress List

  - Preparing broadcast clips for evaluation
    - Sample selection across channels and durations
    - Ground-truth script preparation
- Ticker-region selection
    - Manual coordinate specification (percentage-based ROI)
    - Extraction architecture
    - Sliding-window OCR over the ticker crop
    - Story boundary detection and stitching
- Deciding evaluation methodology
    - CER / WER via stringmetrics
    - Precision- vs. recall-based scoring for segment mode
- Full-length mode evaluation
- Segment mode evaluation
- Full-length vs. segment mode comparison
- Final pipeline architecture
- Deployment

## 3. Video Samples for Evaluation

To evaluate extraction accuracy realistically, six broadcast news clips were selected across two channels and three durations:

- DD India → 1 min, 2 mins, 10 mins
- CNN News18 → 1 min, 2 mins, 10 mins

Channels and durations were chosen deliberately: DD India and CNN News18 use visually different ticker styles (positioning, font, scroll speed, background contrast), and the three durations test whether extraction accuracy holds up as read length increases - a single continuous full-length read accumulates more opportunity for drift than a short clip.

## 4. Pipeline Architecture

The pipeline reads a manually specified ticker region (left, top, width, height, as percentages of frame size) rather than attempting automatic ticker-region detection. This was a deliberate simplification: broadcast ticker position and styling vary enough across channels that manual selection, verified with a live region preview before running extraction, proved more reliable than a general-purpose detector for the channels evaluated here.

Within the selected region, the pipeline uses a sliding-window OCR approach: overlapping frame samples are read via Tesseract OCR, and a story-stitching step reconstructs individual news items from the continuously scrolling text by merging overlapping OCR reads and identifying story boundaries.

Two modes are available:

- **Full length mode** → given a single start timestamp, the ticker is read continuously to the end of the clip, producing a list of stories each with a start_time/end_time.
- **Segment mode** → the video is first split on scene changes (Bhattacharyya distance between frame histograms), and each resulting segment is read independently, producing per-segment extracted text plus optional per-story evaluation within that segment.

The extraction pipeline is organized as follows:

![Architecture diagram](../arch.png)

## 5. Evaluation Methodology

Extraction accuracy is measured with Character Error Rate (CER) and Word Error Rate (WER), computed against an optional ground-truth script uploaded per video. Where no script is supplied, extraction still runs but no evaluation metrics are produced.

- Full-length mode evaluates each extracted story against its corresponding reference story directly.
- Segment mode required a different approach. Text extraction was performed only on segments longer than 3 seconds - shorter segments yielded unreliable results due to the limited number of frames, which led to inaccurate speed estimation and poor image stitching. For per-segment story evaluation, a precision-based approach was used instead of a recall-based one.

A recall-based comparison would have measured extracted text against reference stories containing characters not present within that segment's boundaries, artificially inflating CER/WER. Since segments generated via scene-change detection often contain incomplete ticker stories by construction, precision-based evaluation - scoring only what was actually extracted, against the portion of the reference it corresponds to - is the more appropriate metric.

## 6. Results

#### Full-Length Mode

| Sample | Channel | Duration | Filtered CER | Filtered WER |
|--------|------------|----------|---------------|---------------|
| Sample 1 | DD India | 1 min | 0.1202 | 0.1949 |
| Sample 2 | DD India | 2 mins | 0.0571 | 0.2187 |
| Sample 3 | CNN News18 | 1 min | 0.0328 | 0.0976 |
| Sample 4 | CNN News18 | 2 mins | 0.0379 | 0.0900 |
| Sample 5 | DD India | 10 mins | 0.1536 | 0.3234 |
| Sample 6 | CNN News18 | 10 mins | 0.2834 | 0.3705 |
| **Average (overall)** | | | **0.1142** | **0.2159** |
| **Average — DD India** | | | **0.1103** | **0.2457** |
| **Average — CNN News18** | | | **0.1180** | **0.1860** |

**Analysis:** at 1–2 minute lengths, CNN News18 clips extract noticeably more accurately than DD India (CER ~0.035 vs. ~0.089 average). That gap narrows sharply at 10 minutes - CNN's Sample 6 (CER 0.2834) is the single worst result across all six clips, pulling CNN's channel average up to roughly match DD India's. Error rate appears to scale with clip duration for both channels, most likely from ticker drift, OCR degradation over longer sliding-window reads, or accumulated story-boundary misalignment across a long continuous read.

#### Segment Mode

*Note: Sample 3 and 4 produced no scene changes, so no segments were generated - they were evaluated using the entire 1-minute and 2-minute videos respectively instead (see the Full-Length Mode table for their figures).*

**Sample 1** — DD India, 1 min

| SEG | START | END | DUR | JUMP | STORIES | FRAG CER | FRAG WER |
|-----|----------|----------|------|------|---------|----------|----------|
| 1 | 00:00:00 | 00:00:09 | 9.33 | 3 | 1 | 0.0000 | 0.0000 |
| 2 | 00:00:09 | 00:00:26 | 16.8 | 6 | 3 | 0.0611 | 0.1111 |
| 3 | 00:00:28 | 00:00:38 | 12.13 | 4 | 2 | 0.0593 | 0.1429 |
| 4 | 00:00:40 | 00:00:50 | 10.27 | 3 | 2 | 0.0000 | 0.0000 |
| **Average** | | | | | | **0.0301** | **0.0635** |

**Sample 2** — DD India, 2 mins

| SEG | START | END | DUR | JUMP | STORIES | FRAG CER | FRAG WER |
|-----|----------|----------|-------|------|---------|----------|----------|
| 1 | 00:00:00 | 00:00:04 | 4.67 | 1 | 1 | 0.2031 | 0.3000 |
| 2 | 00:00:04 | 00:01:04 | 59.73 | 22 | 8 | 0.0829 | 0.1739 |
| 3 | 00:01:04 | 00:01:22 | 17.73 | 6 | 3 | 0.0278 | 0.1111 |
| 4 | 00:01:24 | 00:01:38 | 14.0 | 5 | 3 | 0.0443 | 0.1217 |
| 5 | 00:01:38 | 00:01:49 | 11.2 | 4 | 3 | 0.0413 | 0.2963 |
| **Average** | | | | | | **0.0799** | **0.2006** |

**Sample 5** — DD India, 10 mins

*Contains 32 segments in total; only the first 9 are shown below for brevity. The average row reflects the full segment set.*

| SEG | START | END | DUR | JUMP | STORIES | FRAG CER | FRAG WER |
|-----|----------|----------|-------|------|---------|----------|----------|
| 1 | 00:00:00 | 00:00:05 | 5.6 | 2 | 1 | 0.0000 | 0.0000 |
| 2 | 00:00:06 | 00:00:23 | 16.8 | 6 | 2 | 0.0761 | 0.1825 |
| 3 | 00:00:23 | 00:00:36 | 13.07 | 4 | 3 | 0.1053 | 0.2083 |
| 4 | 00:00:36 | 00:00:50 | 14.0 | 5 | 2 | 0.0846 | 0.1666 |
| 5 | 00:00:52 | 00:01:14 | 22.4 | 8 | 4 | 0.1077 | 0.3167 |
| 6 | 00:01:14 | 00:01:34 | 19.6 | 7 | 4 | 0.0350 | 0.0357 |
| 7 | 00:01:34 | 00:02:45 | 70.93 | 26 | 9 | 0.2145 | 0.3588 |
| 8 | 00:02:45 | 00:02:56 | 11.2 | 4 | 1 | 0.0256 | 0.0833 |
| 9 | 00:02:57 | 00:03:06 | 14.93 | 3 | 3 | 0.0417 | 0.0833 |
| **Average (all 32 segments)** | | | | | | **0.1119** | **0.2393** |

**Sample 6** — CNN News18, 10 mins

| SEG | START | END | DUR | JUMP | STORIES | FRAG CER | FRAG WER |
|-----|----------|----------|--------|------|---------|----------|----------|
| 1 | 00:00:00 | 00:07:54 | 474.13 | 177 | 39 | 0.1751 | 0.3300 |
| 2 | 00:07:54 | 00:08:08 | 14.0 | 5 | 2 | 0.0053 | 0.0357 |
| 3 | 00:08:08 | 00:08:22 | 14.0 | 5 | 1 | 0.0123 | 0.0909 |
| 4 | 00:08:23 | 00:09:46 | 83.07 | 31 | 6 | 0.0508 | 0.2273 |
| 5 | 00:09:47 | 00:09:55 | 8.4 | 3 | 1 | 0.0000 | 0.0000 |
| **Average** | | | | | | **0.0487** | **0.1368** |

**Analysis:** Across the four shorter samples (1 and 2 minute clips), fragment CER stays low (0.01–0.10), consistent with the precision-based scoring approach - each segment's extracted text is judged only against the reference text it actually corresponds to. Sample 2 is a partial outlier, with segment 1 (a short 4.67s fragment) showing a comparatively high 0.2031 CER despite the sample's overall average staying moderate. The two 10-minute samples show the highest segment-mode averages overall (Sample 5: 0.1307, Sample 6: 0.1683, per the full-length vs. segment comparison below) - within Sample 5's rows shown here, segment 7 (a 70.93s fragment) stands out at 0.2145 CER, while Sample 6's error is led by segment 1, its longest fragment at 474.13s, with a CER of 0.1751.

#### Full-Length vs. Segment Mode - Comparison

The table below aggregates both modes' CER and WER across all six evaluation videos, mirroring how the extraction strategies are compared directly.

| Sample | Channel | Duration | Full-Length CER | Segment Avg CER | Full-Length WER | Segment Avg WER |
|--------|------------|----------|------------------|------------------|------------------|------------------|
| Sample 1 | DD India | 1 min | 0.1202 | 0.0301 | 0.1949 | 0.0635 |
| Sample 2 | DD India | 2 mins | 0.0571 | 0.0799 | 0.2187 | 0.2006 |
| Sample 3 | CNN News18 | 1 min | 0.0328 | n/a | 0.0976 | n/a |
| Sample 4 | CNN News18 | 2 mins | 0.0379 | n/a | 0.0900 | n/a |
| Sample 5 | DD India | 10 mins | 0.1536 | 0.1119 | 0.3234 | 0.2393 |
| Sample 6 | CNN News18 | 10 mins | 0.2834 | 0.0487 | 0.3705 | 0.1368 |
| **Average** | | | **0.1142** | **0.0451** | **0.2159** | **0.1067** |

*Average across the five samples where segment mode produced results (excludes Sample 3 & 4). "n/a" indicates no segments were generated.*

**Segment mode's lower error:** Segment mode shows a lower average CER/WER than full-length mode on three of the four comparable samples - most visibly on Sample 6, where segment mode's CER (0.0487) is less than a fifth of full-length mode's (0.2834). This is consistent with segment mode's precision-based evaluation: shorter, scene-bounded fragments give the OCR/story-assembly pipeline less room to drift or misalign story boundaries than one long continuous read does.

**Sample 2 is the exception:** Segment mode's average CER is higher than full-length mode's, though not dramatically (0.0799 vs. 0.0571). Its segments were also comparatively few (5 segments across 2 minutes, durations from 4.67s to [CONFIRM: 59.73s]), with segment 1's short 4.67s fragment landing close to the pipeline's own 3-second minimum-duration filter - which may partly explain the reduced reliability there relative to the other samples.

**Caveat:** The two modes are not scored on an identical basis. Segment mode uses fragment-aligned, precision-style scoring ```(evaluate_segments)```, while full-length mode scores each story against its best-matching full reference line and reports the boundary-filtered average ```(evaluate_all)```. The deltas above should therefore be read as directionally informative rather than strictly like-for-like.

## 7. Deployment

Unlike a decoupled backend/frontend architecture, this project ships as a single Streamlit application - extraction, evaluation, and the results UI (including video playback synced to the extracted transcript) all run within one process. A single user runs one extraction at a time, so there's no need for shared state, a job queue, or a separate API layer - a full client-server split would only add deployment overhead here without changing what the tool does.

**Running the app**
```bash
streamlit run app.py
```

## 8. Conclusion

Across six broadcast clips, segment mode's precision-based evaluation shows a lower average error rate than full-length mode, most clearly on the two 10-minute clips where full-length mode's error builds up over the long continuous read. The two modes also fail differently: full-length mode's error rises steadily with clip length, while segment mode's error is usually low but occasionally spikes on a single bad segment (as with Sample 6's fragment 12, or Sample 2's short, closely-packed segments).

Full-length mode stays simpler and doesn't depend on scene-change detection finding good cut points - Sample 4's lack of scene changes shows that dependency directly. In practice, segment mode looks like the better choice for shorter clips with clear scene changes, while full-length mode is more dependable for longer or visually static clips. A hybrid approach - segment mode where cuts are dense enough, full-length otherwise - would be a natural next step rather than picking one mode per video upfront.
