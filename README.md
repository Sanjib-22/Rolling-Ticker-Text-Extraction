# Rolling Ticker Extraction from Broadcast Video

*This project was developed during an internship at EICT IIT Guwahati under the guidance of Prithwijit Guha sir.*

This repository contains an end-to-end pipeline for reading scrolling news-ticker text out of broadcast video — manual ticker-region selection, two extraction modes (full-length and scene-segmented), OCR via Tesseract, accuracy evaluation against a ground-truth script, and a Streamlit web interface with synced video/transcript playback.

[Progress_Report_Final.pdf](docs/Rolling_Ticker_Extraction_Report.pdf) - Progress Report of this project.

[Progress_Report_Notion](https://app.notion.com/p/Rolling-Ticker-Extraction-from-Broadcast-Video-96928754c2ce465db14f204e500944f4?source=copy_link) - Notion Report link.

[Github Page](https://sanjib-22.github.io/Rolling-Ticker-Text-Extraction/) - Github page for this project.

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

## Performance & Evaluation

### Extraction Accuracy (Character/Word Error Rate)

#### Full length mode

| **Video** | **Channel** | **Video Length** | **CER %** | **WER %** |
|--|--|--|--|--|
| sample_1 | DD News | 1m 0s | **12.02%** | **19.49%** |
| sample_2 | DD News | 2m 1s | **5.71%** | **21.87%** |
| sample_1 | CNN-News18 | 1m 0s | **3.28%** | **9.76%** |
| sample_2 | CNN-News18 | 1m 53s | **3.79%** | **9.00%** |
| sample_1 | DD News | 9m 59s | **15.36%** | **32.34%** |
| sample_2 | CNN-News18 | 10m 0s | **28.34%** | **37.05%** |

#### Segmentize mode 

| **Video** | **Channel** | **Video Length** | **Segments** | **Avg. Fragment CER %** | **Avg. Fragment WER %** |
|--|--|--|--|--|--|
| sample_1 | DD News | 1m 0s | 4 | **3.01%** | **6.35%** |
| sample_2 | DD News | 2m 1s | 8 | **7.99%** | **20.06%** |
| sample_3 | CNN-News18 | 1m 0s | 0 | **-** | **-** |
| sample_4 | CNN-News18 | 1m 53s | 0 | **-** | **-** |
| sample_5 | DD News| 9m 59s | 0 | **11.19%** | **23.93%** |
| sample_6 | CNN-News18 | 10m 0s | 0 | **4.87%** | **13.68%** |

---

## Project Structure

```text
Rolling_ticker/
├─ .vscode/
│  └─ settings.json
├─ docs/
│  ├─ Rolling_Ticker_Extraction_Report.pdf
│  └─ sample_coordinates.md
├─ pipeline/
│  ├─ evaluator.py
│  ├─ frameops.py
│  ├─ helpers.py
│  ├─ slidingreader.py
│  ├─ stringmetrics.py
│  ├─ tesseract.py
│  ├─ video_segmentor.py
│  └─ video.py
├─ scripts/
│  ├─ broadcast_summarizer.py
│  ├─ easyrun.py
│  ├─ get_coordinates.py
│  └─ run_on_segments.py
├─ .gitignore
├─ app.py
├─ LICENSE
├─ README.md
└─ requirements.txt               
```

---

## Author

[Sanjib Das](https://github.com/Sanjib-22)

## License

MIT License






