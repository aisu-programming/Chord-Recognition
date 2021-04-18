import numpy as np
import pprint

def overlap(min1, max1, min2, max2):
  return max(0, min(max1, max2) - max(min1, min2))

def BPM_selector(tempo_est):
  if tempo_est.shape[0] >=2:
    tmp = [
      [*tempo_est[0]], 
      [*tempo_est[1]]
    ]
    for i in range(2, tempo_est.shape[0]):
      if abs(tempo_est[i][0] - tmp[0][0]) < abs(tempo_est[i][0] - tmp[1][0]):
        tmp[0][1] += tempo_est[i][1]
      else:
        tmp[1][1] += tempo_est[i][1]
    if tmp[1][1] < tmp[0][1]:
      return tmp[0][0]
    else:
      return tmp[1][0]
  else:
    return tempo_est[0][0]

class PostProcessor():
  def __init__(self, diff_root_only=False, max_num_chord=None):
    self.diff_root_only = diff_root_only
    self.max_num_chord = max_num_chord
    
  def __call__(self, predict_list, bar_list, audio_duration, BPM):
    self.predict_list = predict_list
    self.bar_list = bar_list
    self.bt_list = []
    self.audio_duration = audio_duration
    self.BPM = BPM
    self.sheet = []
    self.first_beat_not_N_time = None
    self.chord_info_list = []

    self._init()
    self._clean_cross_chord_between_measure()
    
    if self.diff_root_only is True:
      self._eliminate_jumping_chord_in_measure()
    
    if self.max_num_chord is not None and isinstance(self.max_num_chord, int):
      self._n_chord_in_measure(self.max_num_chord)    
    
    final_sheet = self._generate_final_sheet()
    
    return self.first_beat_not_N_time, final_sheet

  def _init(self):
    self._init_beats_by_bar()
    self._init_sheet()
    self._build_dominant_list()

  def _init_beats_by_bar(self):
    est_bpm = len(self.bar_list[:, 0]) / (self.audio_duration / 60)
    if est_bpm / self.BPM > 1.8:
      beat_step = 2
    else:
      beat_step = 1

    _, first_chord_time, _ = self._get_first_chord_time()
    
    first_chord_w_beat_idx = 0
    # find the first beat that have chord
    for i in range(self.bar_list.shape[0]):
      if first_chord_time < self.bar_list[i][0]:
        if i == 0:
          first_chord_w_beat_idx = i
        else:
          first_chord_w_beat_idx = i - 1
        break
   
    if int(self.bar_list[first_chord_w_beat_idx][1]) != 1:
      current_beat = int(self.bar_list[first_chord_w_beat_idx][1])
      if first_chord_w_beat_idx > current_beat - 1:
        # backward beat is enough, add N beat to complete the bar
        first_chord_w_beat_idx -= current_beat - 1
      else:
        # backward beat is not enough, discard these beats
        first_chord_w_beat_idx += 5 - current_beat
    
    self.bt_list = self.bar_list[first_chord_w_beat_idx::beat_step, 0]
    self.first_beat_not_N_time = self.bt_list[0]

  def _smooth_beats(self):
    new_beats = []
    temp = []
    start_time = self.bt_list[0]
    for b in self.bt_list:
      if len(temp) != 4:
        temp.append(b)
      else:
        end_time = b
        length = end_time - start_time
        step = length / len(temp)
        for i in range(len(temp)):
          new_beats.append(start_time + i * step)
        start_time = b
        temp = [b]
    for b in temp:
      new_beats.append(b)
    self.bt_list = np.array(new_beats)

  def _init_sheet(self):
    # beat tracking + voting 
    pre_first_match_idx = 0
    measure = []

    for idx, beat in enumerate(self.bt_list):
      # find the start time and end time of this beat
      measure_begin = beat  
      if idx != len(self.bt_list) - 1:
        measure_end = self.bt_list[idx + 1]
      else:
        measure_end = self.audio_duration
     
      max_length = -1
      dominant_chord = None
      first_match_idx = None
      chord = {}

      for idx in range(pre_first_match_idx, len(self.predict_list)):
        predict = self.predict_list[idx]
        if first_match_idx is None and predict[0] <= measure_begin < predict[1]:
          first_match_idx = idx
          pre_first_match_idx = idx

        length = overlap(measure_begin, measure_end, predict[0], predict[1])
        if length > 0:
          chord_str = predict[4]
          if chord_str in chord:
            # case ABA, need to sum two length of A
            chord[chord_str] = ( chord[chord_str][0] + length, *predict[2:] )
          else:
            chord[chord_str] = ( length, *predict[2:])
          
          if chord[chord_str][0] > max_length:
            max_length = chord[chord_str][0]
            dominant_chord = (first_match_idx, *chord[chord_str][1:])

        elif dominant_chord is not None:
          # dominant_chord is assigned, and length < 0
          # exceed the interval of [measure_begin, measure_end]
          break

      if dominant_chord is not None:
        predict_chord = dominant_chord[:]
      else:
        predict_chord = (first_match_idx, *self.predict_list[first_match_idx][2:])

      measure.append( (measure_begin, measure_end, *predict_chord) )
      if len(measure) == 4:
        self.sheet.append(measure)
        measure = []
    
    if len(measure) > 0:
      self.sheet.append(measure)

  def _build_dominant_list(self):
    for measure in self.sheet:
      self.chord_info_list.append( self._get_dominant_chord(measure) )

  def _clean_cross_chord_between_measure(self):
    # no overlap between last beat/next first beat if two measures are "different"

    last_dominant_chord = None
    last_measure = None
    for idx, (measure, chord_info) in enumerate(zip(self.sheet, self.chord_info_list)):
      dominant_chord = chord_info['dominant'][3]
      if idx == 0:
        last_dominant_chord = dominant_chord
        last_measure = measure
        continue
      
      if len(measure) >= 2 and last_measure[-1][5] == measure[0][5] and last_dominant_chord != dominant_chord:
        if last_measure[-1][5] == dominant_chord:
          # case AAAB # BBBB
          last_measure[-1] = (*last_measure[-1][0:2], *last_measure[-2][2:])
        elif measure[0][5] == last_dominant_chord:
          # case AAAA # ABBB
          measure[0] = (*measure[0][0:2], *measure[1][2:])
        
      last_dominant_chord = dominant_chord
      last_measure = measure

  def _eliminate_jumping_chord_in_measure(self):
    # same measure only jump between different roots

    def detect_jump(measure):
      '''
        jump = {
          root : set([(root, quality, chord_str), (root, quality, chord_str)])
          ...
        }
      '''
      jump = {}
      is_jump = False
      for beat in measure:
        root = beat[3]
        if root not in jump:
          jump[root] = set([beat[3:]])
        else:
          jump[root].add(beat[3:])
          if len(jump[root]) > 1:
            is_jump = True
      return is_jump, jump

    for idx, (measure, chord_info) in enumerate(zip(self.sheet, self.chord_info_list)):
      is_jump, jump = detect_jump(measure)
      if is_jump is True:
        # figure out how to replace the chord of the same root
        mapping = {}
        for root, chords in jump.items():
          if len(chords) < 2:
            continue
          
          # in the case of same root, find out which chord is the longest
          max_length = -1
          max_chord = None
          for chord in chords:
            length = chord_info[chord[2]][0]
            if length > max_length:
              max_length = length
              max_chord = chord
          mapping[root] = max_chord
        
        # replace the chord practically
        for idx, beat in enumerate(measure):
          root = beat[3]
          if root in mapping:
            measure[idx] = ( *beat[0:3], *mapping[root] )
   
  def _n_chord_in_measure(self, n_chord):
    # one measure max n chords
    for idx, (measure, chord_info) in enumerate(zip(self.sheet, self.chord_info_list)):
      chords = set()
      for beat in measure:
        chords.add(beat[3:])

      if len(chords) > n_chord:
        # only keep the chord that is the top-n longest, keep_list store the chord that need to keep
        '''
          keep_list = [(root, quality, chord_str), (root, quality, chord_str)...]
        '''
        keep_list = []
        for chord in chords:
          keep_list.append( chord_info[chord[2]] )
        keep_list.sort(key=lambda x:x[0], reverse=True)
        keep_list = [ i[1:] for i in keep_list[:n_chord] ]

        # replace the chord practically
        for idx, beat in enumerate(measure):
          chord = beat[3:]
          if chord in keep_list:
            continue
          else:
            # find the nearest kept chord to replace, priority
            in_range = lambda x : 0 <= x < len(measure)
            
            # priority : 'backward > foreward' if is the first two beat, 'foreward > backward' otherwise
            if idx <= 1 :
              direction = -1
            else:
              direction = 1

            for offset in range(1, len(measure)):
              replace_idx = idx + direction * offset
              if in_range(replace_idx) and measure[replace_idx][3:] in keep_list:
                break
              direction *= -1
              replace_idx = idx + direction * offset
              if in_range(replace_idx) and measure[replace_idx][3:] in keep_list:
                break 
            measure[idx] = (*measure[idx][:3], *measure[replace_idx][3:])

  def _get_first_chord_time(self):
    for idx, predict in enumerate(self.predict_list):
      if predict[4] != 'N':
        return idx, predict[0], predict[1]

  def _get_dominant_chord(self, measure):
    measure_begin = measure[0][0]
    measure_end = measure[-1][1]

    start_idx = measure[0][2]
    max_length = -1
    dominant_chord = None
    chord = {}

    for idx in range(start_idx, len(self.predict_list)):
      predict = self.predict_list[idx]
      length = overlap(measure_begin, measure_end, predict[0], predict[1])
      
      if dominant_chord is not None and length <= 0:
        # exceed the overlap interval
        break

      chord_str= predict[4]
      if chord_str in chord:
        # case of ABAB, need to sum two A's length
        chord[chord_str] = ( chord[chord_str][0] + length, *predict[2:5] )
      else:
        chord[chord_str] = (length, *predict[2:5])
      
      if chord[chord_str][0] > max_length:
        max_length = chord[chord_str][0]
        dominant_chord = chord[chord_str]

    chord['dominant'] = dominant_chord
    
    return chord

  def _generate_final_sheet(self):
    final_sheet = ['#']
    for measure in self.sheet:
      last_chord = None
      for idx, beat in enumerate(measure):
        current_chord = beat[5]
        if last_chord == current_chord:
          final_sheet.append('0')
        else:
          final_sheet.append(current_chord)
          last_chord = current_chord
      
      # padding 0 if measure is not complete
      if len(measure) != 4:
        for i in range( 4 - len(measure) ):
          final_sheet.append('0')

      final_sheet.append('#')
    return final_sheet
