import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from numpy.typing import NDArray
import warnings
warnings.filterwarnings('ignore')


class VideoSegmentation:
    def __init__(self, path : str, no_of_bins : int = 16, frame_skip : int = 28, threshold : float = 0.5):
        self.path = path
        self.no_of_bins = no_of_bins
        self.frame_skip = frame_skip
        self.threshold = threshold

    @staticmethod
    def histogram_maker(frame: NDArray, bin_width: int, no_of_bins: int) -> NDArray:
        hist = [[[] for _ in range(no_of_bins)] for _ in range(3)]
        flat = frame.reshape(-1, 3)

        for pixel in flat:
            b, g, r = pixel

            b_bin = int(np.floor(b / bin_width))
            g_bin = int(np.floor(g / bin_width))
            r_bin = int(np.floor(r / bin_width))

            b_bin = min(b_bin, no_of_bins - 1)
            g_bin = min(g_bin, no_of_bins - 1)
            r_bin = min(r_bin, no_of_bins - 1)

            hist[0][b_bin].append(b)
            hist[1][g_bin].append(g)
            hist[2][r_bin].append(r)

        return hist


    @staticmethod
    def histogram(frame : NDArray, no_of_bins : int) -> NDArray:
        pixels = frame.reshape(-1, 3)
        hist, _ = np.histogramdd(
            pixels,
            bins = no_of_bins,
            range=[(0, 256), (0, 256), (0, 256)]
        )
        hist = hist / hist.sum()
        return hist.astype(float)

    @staticmethod
    def calc_bhattacharyya(hist1 : NDArray, hist2 : NDArray) -> float:
        return np.sum(np.sqrt(hist1 * hist2))

    def calc_bhattacharyya_distance(self, hist1 : NDArray, hist2 : NDArray) -> float:
        bc = self.calc_bhattacharyya(hist1, hist2)
        bc = max(bc, 1e-10)
        distance = -np.log(bc)
        return distance
    
    @staticmethod
    def calc_chi_square(hist1: NDArray, hist2: NDArray) -> float:
        eps = 1e-10
        return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))

    @staticmethod
    def calc_correlation(hist1: NDArray, hist2: NDArray) -> float:
        mean1 = np.mean(hist1)
        mean2 = np.mean(hist2)
        numerator = np.sum((hist1 - mean1) * (hist2 - mean2))
        denominator = np.sqrt(np.sum((hist1 - mean1) ** 2) * np.sum((hist2 - mean2) ** 2))
        if denominator == 0:
            return 0
        return numerator / denominator

    def segment_video(self, method : str = 'bhattachrya distance'):
        cap = cv2.VideoCapture(self.path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps

        frame_skip_interval = min(self.frame_skip, original_fps)
        frame_index = 1
        distances = [0]
        segment_frames = []
        prev_hist = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Finished reading video.")
                break

            if frame_index % frame_skip_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                current_hist = self.histogram(frame, self.no_of_bins)

                if prev_hist is None:
                    prev_hist = current_hist
                else:
                    match method:
                        case 'Bhattachrya Distance':
                            dist = self.calc_bhattacharyya_distance(prev_hist, current_hist)
                        case 'Chi Square':
                            dist = self.calc_chi_square(prev_hist, current_hist)
                        case 'Bhattachrya Coeffecient':
                            dist = self.calc_bhattacharyya(prev_hist, current_hist)
                        case 'Correlation':
                            dist = self.calc_correlation(prev_hist, current_hist)
                        case _ :
                            print("yo check your method")


                    if dist > self.threshold:
                        segment_frames.append(frame_index)
                        
                    distances.append(dist)
                    prev_hist = current_hist

                

            frame_index += 1
            
        segment_time = [round(i / original_fps, 3) for i in segment_frames]
        cap.release()
        return distances, original_fps, frame_skip_interval, duration, segment_frames, segment_time
    
    def plotting(self, distances, fps, skip, method):
        time_axis = [i * skip / fps for i in range(len(distances))]

        scene_changes = [i for i, d in enumerate(distances) if d > self.threshold]

        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, distances, label=f'{method}', color='blue')
        plt.axhline(y=self.threshold, color='green', linestyle='--', label=f'Threshold = {self.threshold}')
        for i in scene_changes:
            plt.axvline(x=time_axis[i], color='red', linestyle='--', alpha=0.6,
                        label='Scene Change' if i == scene_changes[0] else "")
        plt.xlabel('Time (seconds)')
        plt.ylabel(f'{method}')
        plt.title('Scene Change Detection')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def play_video_with_shots(self, segment_frames, wait_time=30):
        """
        Plays the video and displays shot numbers (alternating background colors)
        in the top-left corner.
        Press 'q' to quit, 'p' to pause/resume.
        """
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        shot_starts = [1]
        shot_starts.extend(segment_frames)          
        shot_ends = segment_frames.copy()
        shot_ends.append(total_frames + 1)         
        
        light_bg = (240, 240, 240)   
        dark_bg = (40, 40, 40)       
        light_text = (0, 0, 0)       
        dark_text = (255, 255, 255)  
        
        frame_idx = 0  
        shot_number = 1
        next_shot_frame = segment_frames[0] if segment_frames else total_frames + 1
        
        paused = False
        print("Playback started. Press 'p' to pause/resume, 'q' to quit.")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1  # current 1-indexed frame number
                
                # Update shot number when crossing a segment boundary
                if frame_idx >= next_shot_frame and shot_number < len(segment_frames) + 1:
                    shot_number += 1
                    if shot_number - 1 < len(segment_frames):
                        next_shot_frame = segment_frames[shot_number - 1]
                    else:
                        next_shot_frame = total_frames + 1
                
                if shot_number % 2 == 0:
                    bg_color = dark_bg
                    text_color = dark_text
                else:
                    bg_color = light_bg
                    text_color = light_text
                
                text = f"Shot {shot_number}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = 15, 20
                w, h = text_w + 20, text_h + 20
                # Draw filled rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
                # Draw text
                cv2.putText(frame, text, (x + 10, y + h - 10), font, font_scale, text_color, thickness)
                
                # Optional: show frame number and time
                time_sec = frame_idx / fps
                info_text = f"Frame: {frame_idx}  Time: {time_sec:.2f}s"
                cv2.putText(frame, info_text, (10, frame.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                cv2.imshow("Video with Shot Numbers", frame)
            
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
        
        cap.release()
        cv2.destroyAllWindows()