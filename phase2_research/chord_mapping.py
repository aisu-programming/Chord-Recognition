'''
Copy and edit from ../baseline/chord_mapping.py
'''

# quality_list = ['', 'min', '7', 'maj7', 'min7', 'aug', 'dim', '5', 'maj6']

mapping_default = {
    ''    : '', 
    'min' : 'min', 
    '7'   : '7', 
    'maj7': 'maj7', 
    'min7': 'min7', 
    'aug' : 'aug', 
    'dim' : 'dim', 
    '5'   : '5', 
    'maj6': 'maj6',
}

mapping_majmin = {
    ''    : '', 
    'min' : 'min', 
    '7'   : '', 
    'maj7': '', 
    'min7': 'min', 
    'aug' : '', 
    'dim' : 'min', 
    '5'   : '', 
    'maj6': '',
}

mapping_seventh = {
    ''    : '',
    'min' : 'min', 
    '7'   : '7', 
    'maj7': 'maj7',
    'min7': 'min7', 
    'aug' : '', 
    'dim' : 'min', 
    '5'   : '', 
    'maj6': '',
}