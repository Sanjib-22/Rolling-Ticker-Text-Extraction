"""
Generates a 20-second synthetic 'news broadcast' clip with a rolling ticker
at the bottom, plus a matching ground-truth script file.
Used only to smoke-test the pipeline + Streamlit app. Not part of the app.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 640, 360
FPS = 25
DURATION = 20
N_FRAMES = FPS * DURATION

BAND_TOP = 322
BAND_BOT = 349          # inclusive
BAND_H = BAND_BOT - BAND_TOP + 1

SPEED = 4.0             # pixels per frame -> 100 px/s

STORIES = [
    "PARLIAMENT CLEARS NEW DIGITAL PRIVACY BILL AFTER LONG DEBATE",
    "MONSOON RAINS EXPECTED TO REACH NORTHERN STATES BY FRIDAY",
    "STOCK MARKETS CLOSE HIGHER FOR THE THIRD STRAIGHT SESSION",
    "NATIONAL TEAM ANNOUNCES SQUAD FOR THE UPCOMING TEST SERIES",
]

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 20
GAP = 140               # pixel gap between stories


def build_strip():
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    probe = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(probe)

    widths = [int(d.textlength(s, font=font)) for s in STORIES]
    total = sum(widths) + GAP * len(STORIES)

    strip = Image.new("RGB", (total, BAND_H), (12, 20, 60))
    ds = ImageDraw.Draw(strip)
    x = 0
    for s, w in zip(STORIES, widths):
        ds.text((x, 3), s, font=font, fill=(255, 255, 255))
        x += w + GAP
    return np.array(strip)[:, :, ::-1].copy()   # RGB -> BGR


def scene_background(frame_idx):
    """Three visually distinct scenes so the segmentor has real cut points."""
    t = frame_idx / FPS
    if t < 7.0:
        base = np.zeros((H, W, 3), np.uint8)
        base[:, :] = (30, 45, 70)
        cv2.rectangle(base, (60, 60), (580, 300), (55, 80, 120), -1)
        cv2.circle(base, (320, 150), 55, (200, 205, 215), -1)
        cv2.rectangle(base, (250, 205), (390, 300), (40, 40, 45), -1)
    elif t < 13.5:
        base = np.zeros((H, W, 3), np.uint8)
        base[:, :] = (150, 120, 40)
        for i in range(0, W, 40):
            cv2.line(base, (i, 0), (i - 120, H), (190, 160, 70), 6)
        cv2.rectangle(base, (140, 90), (500, 260), (240, 240, 235), -1)
    else:
        base = np.zeros((H, W, 3), np.uint8)
        base[:, :] = (35, 110, 55)
        cv2.ellipse(base, (320, 190), (220, 110), 0, 0, 360, (60, 150, 80), -1)
        cv2.rectangle(base, (280, 120), (360, 260), (230, 230, 230), -1)
    return base


def main():
    strip = build_strip()
    strip_w = strip.shape[1]
    strip2 = np.concatenate([strip, strip], axis=1)   # allow wraparound

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("test_ticker.mp4", fourcc, FPS, (W, H))

    for i in range(N_FRAMES):
        frame = scene_background(i)

        # ticker band background
        cv2.rectangle(frame, (0, BAND_TOP - 6), (W, H), (12, 20, 60), -1)
        cv2.rectangle(frame, (0, BAND_TOP - 8), (W, BAND_TOP - 6), (0, 90, 220), -1)

        offset = int(i * SPEED) % strip_w
        window = strip2[:, offset:offset + W]
        frame[BAND_TOP:BAND_BOT + 1, 0:W] = window

        out.write(frame)

    out.release()

    with open("test_ticker_script.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(STORIES) + "\n")

    print(f"Wrote test_ticker.mp4  ({DURATION}s, {W}x{H}, {FPS}fps)")
    print(f"Ticker band: y={BAND_TOP}-{BAND_BOT}  ({BAND_H}px)")
    print(f"Coordinates: left=0.0 top={BAND_TOP/H:.3f} "
          f"width=1.0 height={BAND_H/H:.3f}")


if __name__ == "__main__":
    main()