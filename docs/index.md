# Rolling Ticker Extraction from Broadcast Video

*This project was developed during an internship at EICT IIT Guwahati under the guidance of Prithwijit Guha sir.*

**[Read the Full Project Report Here](report.md)** - Browse the detailed extraction methodology, evaluation results, and pipeline architecture.

**[View Source Code on GitHub](https://github.com/Sanjib-22/Rolling-Ticker-Text-Extraction)** - Access the complete repository, pipeline modules, and CLI scripts.

## Project Demonstration

<iframe width="100%" style="aspect-ratio:16/9;" src="https://youtu.be/83neoKqOx9A" title="Rolling Ticker — project walkthrough" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

## Environment Setup

The project requires **Python 3.12**. Once your Python environment is ready, install the required dependencies:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows (PowerShell)
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Tesseract OCR must also be installed and available on your system `PATH` — the pipeline shells out to it directly, it isn't installed via pip:

```bash
# Windows
https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Sample Videos & Ticker Coordinates

To find the ticker's exact region (as percentages of frame size) for a given channel, run:

```bash
.\venv\Scripts\Activate.ps1
python scripts/get_coordinates.py samples/your_video.mp4
```

Reference coordinates for known channels are kept in [`docs/sample_coordinates.txt`](https://github.com/PLACEHOLDER_USER/PLACEHOLDER_REPO/blob/main/docs/sample_coordinates.txt) in the repository. To generate a synthetic test clip instead of using a real broadcast video:

```bash
python scripts/make_test_video.py
```

Videos and generated master word lists live under `samples/`, created automatically on first run.

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

The project includes a Streamlit web app to run extraction interactively, preview the ticker region, and watch results synced to video playback. **Make sure to activate the necessary Python environment before running this**.

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
├── app.py                 Streamlit app — entry point
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
├── samples/                gitignored — runtime output, created automatically
├── License                 MIT License
└── README.md
```

## Author

[Sanjib Das](https://github.com/Sanjib-22)

## License

MIT License
