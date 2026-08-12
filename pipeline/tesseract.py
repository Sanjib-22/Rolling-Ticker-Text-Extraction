import cv2
import subprocess
import os
import csv
import helpers
from helpers import makedirs, cleandir, requests
from frameops import *

class OCR():
  def __init__(self):
    self._images = []

  def __del__(self):
    del self._images

  def _preprocess(self, image):
    # Adjusts the image so that OCR can better read it
    return image.copy()

  def _preprocess_list(self, images):
    return [self._preprocess(img) for img in images]

  def add(self, images):
    if not type(images) is list:
      images = [images]
    self._images = self._images + self._preprocess_list(images)

  def _runocr(self, images):
    # Takes preprocessed image list, runs OCR, and returns texts
    raise NotImplementedError

  def read(self, images = None, language = None):
    if images is None:
      images = self._images
      self._images = []
    else:
      if not type(images) is list:
        images = [images]
      images = self._preprocess_list(images)
    if len(images) == 0:
      return []
    else:
      return self._runocr(images, language) # type: ignore

# TesseractOCR — feeding from the OCR interface defined above
class TesseractOCR(OCR):
  def __init__(self, tmp_path = './ocr/TesseractOCR/tmp', **preprocesses):
    super(TesseractOCR, self).__init__();
    self.tmp_path = tmp_path
    self.video_id = preprocesses.pop('video_id', 'unknown')
    self.master_tsv_path = preprocesses.pop(
        'master_tsv_path', './samples/master_files/master_words.tsv'
    )
    self._preprocesses = preprocesses
    makedirs(tmp_path + '/input')
    makedirs(tmp_path + '/output')
    makedirs(os.path.dirname(self.master_tsv_path))
  
  def _preprocess(self, image):
    if requests('gamma_correct', self._preprocesses):
      image = gamma_to_intensity(image, **self._preprocesses)
  
    image = hist_adjust(image, **self._preprocesses)
    image = deblur_horizontal(image, **self._preprocesses)
    
    if requests('resize_font', self._preprocesses):
      if 'height' in self._preprocesses:
        image = resize_font(image, **self._preprocesses)
        if 'new_height' in self._preprocesses:
          new_height = self._preprocesses['new_height']
        else:
          new_height = 32
        self.scale_factor = new_height / self._preprocesses['height']
      else:
        raise ValueError('Tried to calculate height')
        image = resize_font(image, height = image.shape[0], **self._preprocesses)
    else:
      self.scale_factor = 1.0
        
    if requests('add_padding', self._preprocesses):
      if 'padding' in self._preprocesses:
        pad = self._preprocesses['padding']
      else:
        pad = 2
      
      edges = np.concatenate([image[:2], image[-2:]], axis = 0)
      if edges[edges < edges.mean() + 0.001].size > edges[edges > edges.mean() - 0.001].size:
        pad_color = edges[edges < edges.mean() + 0.001].mean()
      else:
        pad_color = edges[edges > edges.mean() - 0.001].mean()
      
      image = np.pad(image, ((pad, pad), (0, 0)), mode = 'constant', constant_values = pad_color)
        
    if requests('gamma_correct', self._preprocesses):
      image = gamma_to_rgb(image, **self._preprocesses)
  
    return image
  
  def _append_to_master_tsv(self, tsv_path):
    if not os.path.exists(tsv_path):
      return
 
    file_exists = os.path.exists(self.master_tsv_path)
 
    with open(tsv_path, 'r', encoding='utf-8') as f_in:
      reader = csv.reader(f_in, delimiter='\t', quoting=csv.QUOTE_NONE)
      rows = list(reader)
 
    if not rows:
      return
 
    with open(self.master_tsv_path, 'a', encoding='utf-8', newline='') as f_out:
      writer = csv.writer(f_out, delimiter='\t', quoting=csv.QUOTE_NONE,
                          escapechar='\\')
      if not file_exists:
        writer.writerow(['video_id'] + rows[0])
      
      for row in rows[1:]:
        if any(cell.strip() for cell in row):
          writer.writerow([self.video_id] + row)
  
  def _runocr(self, images, language = None):
    image_list_path = self.tmp_path + '/input' + '/image_list.txt'
    image_list = open(image_list_path, 'w')
    for img_num in range(len(images)):
      name = str(img_num) + '.png'
      image_path = self.tmp_path + '/input' + '/' + name
      images[img_num] = frames_to_int(images[img_num])
      cv2.imwrite(image_path, images[img_num])
      image_list.write(image_path + '\n')
    image_list.close()
    
    if language is None or language == 'English':
      lan = 'eng'
    else:
      raise ValueError('Language ' + language + ' not implemented')
    output_path = self.tmp_path + '/output' + '/words'
    # Run OCR and write read words to Tab Seperated Value file
    subprocess.call(['tesseract', image_list_path, output_path, '-l', lan, '--psm', '6', 'tsv'])
    self._append_to_master_tsv(output_path + '.tsv')
    ocr_output = helpers.read(output_path + '.tsv')
    
    result = []
    for row in ocr_output:
      # Check if new image started
      if row['level'] == '1':
        last_page = row['page_num']
        result.append([])
      # Check if row is a word
      elif (row['level'] == '5') and\
           (len(row['text']) > 0) and\
           (row['text'] != ' '):
        word = {}
        word['text'] = row['text']
        word['left'] = float(row['left']) / self.scale_factor
        word['top'] = float(row['top']) / self.scale_factor
        word['right'] = (float(row['left']) + float(row['width'])) / self.scale_factor
        word['bottom'] = (float(row['top']) + float(row['height'])) / self.scale_factor
        word['confidence'] = float(row['conf']) / 100.0
        result[-1].append(word)
    
    # Clean tmp
    cleandir(self.tmp_path + '/input');
    cleandir(self.tmp_path + '/output');
    return result