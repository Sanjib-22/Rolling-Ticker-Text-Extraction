import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: python get_coordinates.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

# Seek to 10% into video to avoid blank frames
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.1))
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read frame")
    sys.exit(1)

fh, fw = frame.shape[:2]
print(f"Frame size: {fw}x{fh}")
print("Click TOP-LEFT of ticker region, then BOTTOM-RIGHT")
print("Press 'r' to reset, 'q' to quit and calculate")

clicks = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert display coordinates back to original
        actual_x = int(x / display_scale)
        actual_y = int(y / display_scale)
        clicks.append((actual_x, actual_y))
        
        # Draw on display frame
        cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)
        if len(clicks) == 2:
            x1d = int(clicks[0][0] * display_scale)
            y1d = int(clicks[0][1] * display_scale)
            x2d = int(clicks[1][0] * display_scale)
            y2d = int(clicks[1][1] * display_scale)
            cv2.rectangle(frame_display, (x1d, y1d), (x2d, y2d), (0, 255, 0), 2)
        cv2.imshow('Select Ticker Region', frame_display)
        print(f"  Clicked: actual x={actual_x}, y={actual_y}")

frame_copy = frame.copy()
display_scale = 1.0
if fw > 1280:
    display_scale = 1280 / fw
    display_h = int(fh * display_scale)
    display_w = int(fw * display_scale)
    frame_display = cv2.resize(frame_copy, (display_w, display_h))
else:
    frame_display = frame_copy
cv2.imshow('Select Ticker Region', frame_display)
cv2.setMouseCallback('Select Ticker Region', on_click)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        clicks.clear()
        frame_display = cv2.resize(frame.copy(), (display_w, display_h)) if fw > 1280 else frame.copy()
        cv2.imshow('Select Ticker Region', frame_display)
        print("Reset — click again")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

if len(clicks) >= 2:
    x1, y1 = clicks[0]
    x2, y2 = clicks[1]

    # Ensure x1,y1 is top-left
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    left   = round(x1 / fw, 3)
    top    = round(y1 / fh, 3)
    width  = round((x2 - x1) / fw, 3)
    height = round((y2 - y1) / fh, 3)

    print(f"\nCoordinates calculated:")
    print(f"  Pixels : x={x1}-{x2}, y={y1}-{y2}")
    print(f"  left={left} top={top} width={width} height={height}")
    print(f"\nEnter this in easyrun.py:")
    print(f"  {left} {top} {width} {height}")
else:
    print("Need 2 clicks — run again")