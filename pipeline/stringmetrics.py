import numpy as np

def edit_distance_with_errors(str1, str2, debug = False):
    distances = np.zeros(shape = [len(str1) + 1, len(str2) + 1],\
                         dtype = np.int32)
    
    for idx1 in range(0, len(str1) + 1):
        distances[idx1, 0] = idx1
    for idx2 in range(0, len(str2) + 1):
        distances[0, idx2] = idx2
    for idx1 in range(0, len(str1)):
        for idx2 in range(0, len(str2)):
            distances[idx1 + 1, idx2 + 1] = min(\
                distances[idx1, idx2 + 1],\
                distances[idx1 + 1, idx2],\
                distances[idx1, idx2]) + 1
            if str1[idx1] == str2[idx2]:
                distances[idx1 + 1, idx2 + 1] = min(\
                    distances[idx1 + 1, idx2 + 1],\
                    distances[idx1, idx2])
    
    # Backtrack to get error types
    pos = [len(str1), len(str2)]
    insertions = 0
    deletions = 0
    substitutions = 0
    while pos != [0, 0]:
        curx = pos[0]
        cury = pos[1]
        curdist = distances[curx, cury]
        
        if curx == 0:
            if debug:
              print('delb @', pos)
            deletions+= 1
            pos[1]-= 1
        elif cury == 0:
            if debug:
              print('insb @', pos)
            insertions+= 1
            pos[0]-= 1
        elif (distances[curx - 1, cury - 1] == curdist) and (str1[curx - 1] == str2[cury - 1]):
            pass
            pos[0]-= 1
            pos[1]-= 1
        elif (distances[curx - 1, cury - 1] == curdist - 1) and (str1[curx - 1] != str2[cury - 1]):
            substitutions+= 1
            if debug:
              print('sub @ ', pos)
            pos[0]-= 1
            pos[1]-= 1
        elif (distances[curx, cury - 1] == curdist - 1):
            if debug:
              print('del @ ', pos)
            deletions+= 1
            pos[1]-= 1
        elif distances[curx - 1, cury] == curdist - 1:
            if debug:
              print('ins @', pos)
            insertions+= 1
            pos[0]-= 1
    
    res = {}

    res['edit_distance'] = distances[-1, -1]
    res['insertions'] = insertions
    res['substitutions'] = substitutions
    res['deletions'] = deletions
    
    res['error_rate'] = res['edit_distance'] / len(str2)
    res['accuracy'] = 1 - res['error_rate']
            
    return res
    
def edit_distance(str1, str2):
    return edit_distance_with_errors(str1, str2)['edit_distance']

def similarity_metric(str1, str2):
    return 1 - edit_distance(str1, str2) / max(len(str1), len(str2))
    
def split_words(str1):
    result = []
    word = ''
    for char in str1:
        if char != ' ':
            word+= char
        elif char == ' ':
            if len(word) > 0:
                result.append(word)
                word = ''
    
    if word != '':
        result.append(word)
    
    return result
    
def error_rate(str1, str2):
    return edit_distance(str1, str2) / len(str2)
    
def accuracy(str1, str2):
    return 1 - error_rate(str1, str2)