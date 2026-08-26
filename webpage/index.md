# Rolling Ticker Extraction from Broadcast Video

*This project was developed during an internship at EICT IIT Guwahati under the guidance of Prithwijit Guha sir.*

**[Read the Full Project Report Here](https://app.notion.com/p/Rolling-Ticker-Extraction-from-Broadcast-Video-96928754c2ce465db14f204e500944f4?source=copy_link)** - Browse the detailed extraction methodology, evaluation results, and pipeline architecture.

**[View Source Code on GitHub](https://github.com/Sanjib-22/Rolling-Ticker-Text-Extraction)** - Access the complete repository, pipeline modules, and CLI scripts.

## Project Demonstration

> **P.S. Additional notes not mentioned in the video:**
>
> - Ticker region is set manually as percentage coordinates (left/top/width/height), not auto-detected.
> - Text extraction uses Tesseract OCR via a sliding-window read over the ticker crop.
> - Segment mode splits the video on scene changes (Bhattacharyya distance) before reading each segment independently; segments shorter than 3 seconds are skipped as unreliable.

<iframe width="100%" style="aspect-ratio:16/9;" src="https://www.youtube.com/watch?v=6X41XHpkvZ4" title="Rolling Ticker — project walkthrough" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<!--
  Replace YOUR_VIDEO_ID_HERE above once the walkthrough is uploaded:
  1. Upload the screen recording to YouTube as Unlisted (viewable by anyone
     with the link, without appearing in search/your channel listing).
  2. Copy the ID from the URL: https://youtu.be/THIS_PART
  3. Paste it in place of YOUR_VIDEO_ID_HERE above.
-->

---

## Environment Setup

The project requires **Python 3.12**. Once your Python environment is ready, install the required dependencies:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1    
pip install -r requirements.txt
```

Tesseract OCR must also be installed and available on your system `PATH` — the pipeline shells out to it directly, it isn't installed via pip:

```bash
https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Sample Videos & Ticker Coordinates

To find the ticker's exact region (as percentages of frame size) for a given channel, run:

```bash
.\venv\Scripts\Activate.ps1
python scripts/get_coordinates.py samples/your_video.mp4
```

Reference coordinates for known channels are kept in [`docs/sample_coordinates.md`](/docs/sample_coordinates.md).

---

## Running via CLI

The core extraction logic can be run directly, outside the web app:

1. **Full length mode** — reads the ticker from a start timestamp to the end of the clip:
   ```
   python scripts/easyrun.py samples/sample_3.mp4
   ```
2. **Segment mode** — splits the video on scene changes first, then reads each segment independently:
   ```
   python scripts/run_on_segments.py samples/sample_3.mp4
   ```

*Note: extracted text, evaluation metrics, and the accumulated master word list are saved under `samples/master_files/` during these runs.*

---

## Web App Deployment

The project includes a Streamlit web app to run extraction interactively, preview the ticker region, and watch results synced to video playback. **Make sure to activate the necessary Python environment before running this.**

**Start the app**
```bash
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

> Always run from the project root — a few paths (`samples/`, `ocr/TesseractOCR/tmp/`) resolve relative to the current working directory, not the file's location.

---

## Project Structure

```
rolling-ticker/
├── app.py                 Streamlit app 
├── requirements.txt
├── pipeline/               Core extraction pipeline
│   ├── video.py             Video frame access
│   ├── video_segmentor.py   Scene-change based segmentation
│   ├── frameops.py          Frame preprocessing
│   ├── tesseract.py         OCR interface + Tesseract implementation
│   ├── slidingreader.py     Ticker reading, story assembly
│   ├── stringmetrics.py     Text similarity helpers
│   ├── evaluator.py         CER/WER evaluation against ground truth
│   └── helpers.py           Generic no-domain helpers
├── scripts/                 Standalone CLI tools
│   ├── easyrun.py             Full-length CLI run
│   ├── run_on_segments.py     Segment-mode CLI run
│   ├── get_coordinates.py     Find ticker ROI coordinates
│   ├── make_test_video.py     Synthetic test clip
│   └── broadcast_summarizer.py
├── docs/
│   └── sample_coordinates.txt   Example ticker ROI values by channel
├── samples/                Gitignored - runtime output, created automatically
├── License                 
└── README.md
```

## Author

[Sanjib Das](https://github.com/Sanjib-22)

## License

MIT License
