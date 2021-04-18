import re

note_dict = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

"""
Convert a note string to integer temperament.
Return 12 if not valid.
"""

def note_to_temperament(note_str):
    if note_str[0] not in note_dict:
        return 12
    temperament = note_dict[note_str[0]]
    for symbol in note_str[1:]: # for double flat
        if symbol == "#":
            temperament += 1
        elif symbol == "b":
            temperament -= 1
    return temperament % 12

"""
Parse a chord string into temperament, quality, inversion
"""

def parse_chord(chord_str):
    chord_root = ""
    chord_quality = ""
    inversion = ""

    if ":" in chord_str:
        if "/" in chord_str:
            chord_root, chord_quality, inversion = re.split(":|/", chord_str)
        else:
            chord_root, chord_quality = chord_str.split(":")
    else:
        if "/" in chord_str:
            chord_root, inversion = chord_str.split("/")
        else:
            chord_root = chord_str

    return note_to_temperament(chord_root), chord_quality, inversion

"""
Generate chord string given root and quality
"""

def get_chord_str(root, quality, valid_list):
    chord_str = ""
    if root == 12:
        chord_str = "N"
    else:
        chord_str = list(note_dict.keys())[list(note_dict.values()).index(root)]
        if valid_list[quality] != "":
            chord_str += ":"
        chord_str += valid_list[quality]

    return chord_str

"""
Make sure the quality is in the range of interest.
Return 'none' if not valid.
"""

def to_valid_scope(quality, valid_list):
    if quality == "maj" or quality == "1":
        quality = ""
    if quality not in valid_list:
        return "none"
    else:
        return quality
