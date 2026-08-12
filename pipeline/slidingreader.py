import random
from helpers import inrange
from tesseract import TesseractOCR

def get_frame_list(video, jump_size = 6, **kwargs):
    
    # Returns list of frame numbers including first and last frame
    frame_numbers =\
      [frame_number for frame_number in range(0, video.frame_count, jump_size)]
    last_frame_number = video.frame_count - 1;
    if frame_numbers[-1] != last_frame_number:
        frame_numbers.append(last_frame_number)
    
    return frame_numbers

def cut_window(image, window):
    
    # Returns image cropped to window
    return image[window[0][0]: window[0][1] + 1, window[1][0]: window[1][1] + 1].copy()

def get_ticker_images(video, ticker, frame_numbers):
    
    # Returns list of images of ticker from video at frame_numbers
    images = []
    for frame_number in frame_numbers:
        frame = video.frame(frame_number)
        images.append(cut_window(frame, ticker))
    
    return images
    
def estimate_speed(word_lists, frame_numbers, min_speed = 0.05, max_speed = 0.6, height = None, **kwargs):
    
    # Estimate movement speed of ticker, by looking at position of the same word in consecutive frames
    estimate_sum = 0
    estimate_count = 0
    
    # Iterate over every pair of neighbouring frames
    for i in range(len(frame_numbers) - 1):
        word_list1 = word_lists[i]
        word_list2 = word_lists[i + 1]
        
        frame_diff = frame_numbers[i + 1] - frame_numbers[i]
        
        min_diff = min_speed * height * frame_diff # type: ignore
        max_diff = max_speed * height * frame_diff # type: ignore
        
        for word1 in word_list1:
            for word2 in word_list2:
                left_diff = word1['left'] - word2['left']
                right_diff = word1['right'] - word2['right']
                
                if not word1['text'] == word2['text']:
                    continue
                if not inrange(left_diff, min_diff, max_diff):
                    continue
                if not inrange(right_diff, min_diff, max_diff):
                    continue
                if not len(word1['text']) >= 3:
                    continue
                
                pixel_diff = word1['left'] - word2['left']
                estimate = pixel_diff / frame_diff
                estimate_sum+= estimate
                estimate_count+= 1
    
    return estimate_sum / estimate_count
    
def add_frame_numbers(word_lists, frame_numbers):
  for i, frame_number in enumerate(frame_numbers):
    for word in word_lists[i]:
      word['frame'] = frame_number
    
def add_long_positions(word_lists, speed):
    
    # Calculates and adds word position on long text
    for word_list in word_lists:
        for word in word_list:
            word['longleft'] = word['left'] + speed * word['frame']
            word['longright'] = word['right'] + speed * word['frame']
    
def concatenate_word_lists(word_lists):
    result = []
    for word_list in word_lists:
        result += word_list
    
    return result
    
def remove_edge_words(word_list, width = None, height = None, **kwargs):
    
    #Remove words that might be partially out of frame.
    if width != None:
      result = []
      for word in word_list:
        if word['left'] > height and word['right'] < width - height:
          result.append(word)
      return result
    else:
      raise ValueError('No width supplied in testing')
    
def remove_garbage(word_list, garbage_method = 'char_confidence', garbage_threshold = 0.80, height = None, **kwargs):
    
    # Return word_list with incorrectly read words removed, that were likely read incorrectly
    if garbage_method is None:
        return word_list
    elif garbage_method == 'char_confidence':
        
        for word in word_list:
            word['char_confidence'] = (word['confidence'] + 0.005) ** (1 / len(word['text']))

        # Remove words with low average confidence
        result = []
        for word in word_list:
            if word['char_confidence'] > garbage_threshold:
                result.append(word)
        return result
    else:
        raise ValueError('Method ' + garbage_method + ' not implemented.')
    
def clusterize_words(word_list, height = None, **kwargs):
    word_list.sort(key = lambda x: x['longleft'])
    if height == None:
      raise ValueError("Tried to calc height")
      height = - 1
      for word in word_list:
          if len(word['text']) > 3:
              height = word['bottom'] - word['top']
              break
    
    # Find clusters using discrete set union
    cluster_members = [[i] for i in range(len(word_list))]
    for idx, word in enumerate(word_list):
        word['cluster'] = idx
    
    def merge_into(clust1, clust2):
        for i in cluster_members[clust2]:
            cluster_members[clust1].append(i)
            word_list[i]['cluster'] = clust1
        cluster_members[clust2] = []
    
    def connect_words(word1, word2):
        clust1 = word1['cluster']
        clust2 = word2['cluster']
        if clust1 == clust2:
            pass
        else:
            # Merge smaller cluster into larger one for efficiency
            if len(cluster_members[clust1]) <\
               len(cluster_members[clust2]):
                merge_into(clust2, clust1)
            else:
                merge_into(clust1, clust2)
    
    max_distance = height
    
    def similar_words(word1, word2):
        if abs(word['longright'] - word2['longright']) > max_distance:
            return False
        elif len(word['text']) != len(word2['text']):
            return False
        else:
            return True
    
    for i, word in enumerate(word_list):

        # Check words with larger longleft
        j = i
        while j < len(word_list) and\
              abs(word_list[j]['longleft'] - word['longleft']) < max_distance:
            word2 = word_list[j]
            if similar_words(word, word2):
                connect_words(word, word2)
            j+= 1

        # Check words with smaller longleft
        j = i
        while j >= 0 and\
              abs(word_list[j]['longleft'] - word['longleft']) < max_distance:
            word2 = word_list[j]
            if similar_words(word, word2):
                connect_words(word, word2)
            j-= 1

def average_word(word_list):
    result = ''
    word_length = len(word_list[0]['text'])
    for position in range(word_length):
        frequencies = {}
        for word in word_list:
            char = word['text'][position]
            if char in frequencies:
                frequencies[char]+= 1
            else:
                frequencies[char] = 1
        
        # Find the most common character and adding to result
        mx = -1
        mxch = ' '
        for p in frequencies.items():
            if p[1] > mx:
                mxch = p[0]
                mx = p[1]
        result+= mxch
        
    return result
            
def merge_cluster(word_list, merge_method = None):
    def avg(l, key):
        s = 0
        for w in l:
            s+= w[key]
        return s / len(l)
    
    result = {}
    result['longleft'] = avg(word_list, 'longleft')
    result['longright'] = avg(word_list, 'longright')
    result['top'] = avg(word_list, 'top')
    result['bottom'] = avg(word_list, 'bottom')
    result['confidence'] = avg(word_list, 'confidence')
    result['merge_size'] = len(word_list)
    result['minframe'] = min([word['minframe'] for word in word_list])
    result['maxframe'] = max([word['maxframe'] for word in word_list])
    result['cluster'] = word_list[0]['cluster']
    
    if merge_method is None:
        merge_method = 'char_frequency'
    if merge_method == 'char_frequency':
        result['text'] = average_word(word_list)
    elif merge_method == 'confidence':
        max_confidence = -1
        for word in word_list:
          if word['confidence'] > max_confidence:
            max_confidence = word['confidence']
            result['text'] = word['text']
    elif merge_method == 'random':
      result['text'] = word_list[random.randint(0, len(word_list) - 1)]['text']
    else:
        raise ValueError('Method ' + merge_method + ' not implemented.')
        
    return result
            
def merge_words(word_list, merge_method = None, **kwargs):
    for word in word_list:
        word['minframe'] = word['frame']
        word['maxframe'] = word['frame']
        
    if merge_method is None:
        for word in word_list:
            word['merge_size'] = 1
        return word_list
      
    clusterize_words(word_list, **kwargs)
    
    result = []
    word_list.sort(key = lambda word: word['cluster'])
    cluster = []
    current_cluster = - 1
    for word in word_list:
        if len(cluster) == 0:
            current_cluster = word['cluster']
            cluster = [word]
        elif word['cluster'] == current_cluster:
            cluster.append(word)
        else:
            result.append(merge_cluster(cluster, merge_method))
            current_cluster = word['cluster']
            cluster = [word]
    
    return result
    
def choose_non_overlapping_words(word_list, overlap_method = None, height = None, **kwargs):
    word_list.sort(key = lambda word: word['longleft'])
    
    result = []
    
    if overlap_method is None:
        overlap_method = 'hybrid'    
    if overlap_method == 'hybrid':
        for word in word_list:
            word['likelihood'] = (0.005 + word['confidence']) ** (1 / len(word['text'])) * word['merge_size']
    elif overlap_method == 'char_confidence':
        for word in word_list:
            word['likelihood'] = (0.005 + word['confidence']) ** (1 / len(word['text']))
    elif overlap_method == 'confidence':
        for word in word_list:
            word['likelihood'] = word['confidence']
    elif overlap_method == 'length':
        for word in word_list:
            word['likelihood'] = len(word['text'])
    elif overlap_method == 'merges':
        for word in word_list:
            word['likelihood'] = word['merge_size']
    elif overlap_method == 'random':
        for word in word_list:
            word['likelihood'] = random.uniform(0.0, 1.0)
    
    for word in word_list:
        while True:
            if len(result) == 0:
                result.append(word)
                break
            if result[-1]['longright'] - height / 2 > word['longleft']: # type: ignore
                if result[-1]['likelihood'] < word['likelihood']:
                    result.pop()
                else:
                    break
            else:
                result.append(word)
                break
    
    return result
    
def seperate_news_stories(word_list, method = None):

    # Returns a list of news stories as dictionaries    
    news_stories = []
    if len(word_list) != 0:
        height = word_list[0]['bottom'] - word_list[0]['top']
        last_longright = - 100000000
        for word in word_list:
            if word['longleft'] - last_longright > height:
                #Make new newsstory
                news_story = {}
                news_story['text'] = ''
                news_story['minframe'] = 100000000
                news_story['maxframe'] = - 100000000
                news_stories.append(news_story)
            else:
                news_story['text']+= ' '
            news_story['text']+= word['text']
            news_story['minframe'] =\
              min(news_story['minframe'], word['minframe'])
            news_story['maxframe'] =\
              max(news_story['maxframe'], word['maxframe'])
            
            last_longright = word['longright']
    
    return news_stories
    
def add_story_start_end_times(story_list, video):
  for story in story_list:
    story['start_time'] = video.frame_duration * story['minframe']
    story['end_time'] = video.frame_duration * story['maxframe']
    
def read_ticker(video, ticker, ocr, **kwargs):
    frame_numbers = get_frame_list(video, **kwargs)
    images = get_ticker_images(video, ticker, frame_numbers)
    word_lists = ocr.read(images, language = video.language)
    speed = estimate_speed(word_lists, frame_numbers, **kwargs)
    add_frame_numbers(word_lists, frame_numbers)
    add_long_positions(word_lists, speed)
    word_list_unmerged = concatenate_word_lists(word_lists)
    word_list_unmerged = remove_edge_words(word_list_unmerged, **kwargs)
    word_list_merged = merge_words(word_list_unmerged, **kwargs)
    word_list_merged = remove_garbage(word_list_merged, **kwargs)
    word_list_final = choose_non_overlapping_words(word_list_merged, **kwargs)
    news_stories = seperate_news_stories(word_list_final)
    add_story_start_end_times(news_stories, video)
    
    return news_stories

