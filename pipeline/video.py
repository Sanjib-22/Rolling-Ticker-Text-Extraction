"""
Loads video and provide option to seek to a specific timestamp and read frames from it.
"""

import cv2
from frameops import *

class Video():
    def __init__(self, path, ocr = None, language = None):
        video = cv2.VideoCapture(path)
        self.video_capture = video
        self.path = path
        self.frame_rate = video.get(cv2.CAP_PROP_FPS)
        self.frame_duration = 1 / self.frame_rate
        self.frame_count = self._get_frame_count()
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.shape = [self.height, self.width, 3]
        if language is not None:
            self.language = language
        elif ocr is not None:
            self.language = self._guess_language(ocr)
        else:
            self.language = None
    
    def __del__(self):
        del self.video_capture
    
    def seek(self, timestamp_sec: float):
        if timestamp_sec < 0:
            timestamp_sec = 0.0
        ms = timestamp_sec * 1000
        self.video_capture.set(cv2.CAP_PROP_POS_MSEC, ms)
        self.frame_count = self._get_frame_count()
        print(f"Seeked to {timestamp_sec:.1f}s — {self.frame_count} frames remaining")

    def _get_frame_count(self):
        def got_frame(video):
            success, frm = video.read()
            return success
        
        start_pos = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        frame_count = 0
        while got_frame(self.video_capture):
            frame_count+= 1
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
        return frame_count
    
    def frame(self, frame_number = None):
        if frame_number is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        read_success, image = self.video_capture.read()
        image = frames_to_float(image)
        return image
        
    def _test_language(self, language, frame_numbers, ocr):
        frames = [self.frame(frame_number) for frame_number in frame_numbers]
        output = ocr.read(frames, language = language)
        confidence = 0
        for word_list in output:
          for word in word_list:
              confidence += word['confidence'] * len(word['text'])
        return confidence
        
    def _guess_language(self, ocr):
        return 'English'