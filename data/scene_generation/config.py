"""Config file
"""
from audiomentations import PitchShift, TimeStretch, SevenBandParametricEQ, ApplyImpulseResponse
import os

# Local dirs to put assets (event mapping and roots dict for events of interest)
ASSETS_PATH = os.path.abspath(os.path.join('../../externals/salt', 'assets'))

IMPULSES_EVENT_MAP_FP = os.path.join(ASSETS_PATH, 'impulses_mapping.tsv')
IMPULSES_ROOT_FP = os.path.join(ASSETS_PATH, 'impulses_roots.json')

SCENES_MAP_FP = os.path.join(ASSETS_PATH, 'scene_mapping.tsv')
SCENES_ROOT_FP = os.path.join(ASSETS_PATH, 'scene_roots.json')


### General generation parameters

audio_duration = 5 #s

# Number of hours for each dataset
train_hours = 50
val_hours = 20
test_hours = 10

GENERATION_PARAMS = {
  "n_mix": {
    "train": int(train_hours * 3600 / audio_duration),
    "val": int(val_hours * 3600 / audio_duration),
    "test": int(test_hours * 3600 / audio_duration)
  },
  "audio_duration": audio_duration,
  "impulse_max_duration": 2.0,
  "n_backgrounds": 1,
  "sr": 44100,
}

TYPE_OF_IMPULSES = "mix"   # "natural", "synthetic", "mix"

N_IMPULSES_DISTRIB = {
    "min_val": 0, # doit être rare
    "max_val": 5,
    "mean": 3,
    "std": 2
  }

# SALT events to filter
# IMPULSE_EVENTS = [
#   'bell',
#   'can_opening',
#   'clapping',
#   'clock_tick',
# #  'cooking',
#   'coughing',
#   'cupboard_open_or_close',
#   'door_knock',
#   'door_wood_creak',
#   'drawer_open_or_close',
#   'explosion',
#   'fireworks',
#   'glass_break',
#   'keyboard_typing',
#   'mouse_click',
#   'object_fall',
#   'object_impact',
#   'switch_on_or_off',
#   'thunderstorm',
#   'typing',
#   'window_opens_or_closes',
#   'synthetic_impulse'  
# ]

# SALT scenes to filter
BACKGROUND_SCENES = [
  'car_interior',
  'bus_interior',
  'train_interior',
  'pedestrian_street_or_square',
  'busy_street',
  'airport',
  'park',
  'shopping_mall',
  'metro_station',
  'truck_interior',
  'cafe_or_restaurant',
  'open-air_market',
  'station_hall',
  'shop',
  'construction_site',
  'bar_nightclub_or_concert_hall',
#  'billiard_pool_hall',
  'kid_game_hall',
  'quiet_street',
  'student_hall',
  'airplane_interior',
  'church',
  'office',
  'train_station',
  'street_balcony',
  'home',
  'living_room',
  'library',
  'synthetic_stationary',
]


# SNR distribution between impulses/backgrounds
SNR_RANGE = {
  'car_interior': {'mean': 3, 'std': 5, 'min_val': -5, 'max_val': 10},
  'bus_interior': {'mean': 1, 'std': 6, 'min_val': -5, 'max_val': 8},
  'train_interior': {'mean': 0, 'std': 5, 'min_val': -5, 'max_val': 7},
  'pedestrian_street_or_square': {'mean': 3, 'std': 7, 'min_val': -4, 'max_val': 10},
  'busy_street': {'mean': 0, 'std': 8, 'min_val': -8, 'max_val': 8},
  'airport': {'mean': 0, 'std': 6, 'min_val': -6, 'max_val': 7},
  'park': {'mean': 5, 'std': 8, 'min_val': -2, 'max_val': 15},
  'shopping_mall': {'mean': 3, 'std': 6, 'min_val': -5, 'max_val': 10},
  'metro_station': {'mean': 1, 'std': 6, 'min_val': -6, 'max_val': 8},
  'truck_interior': {'mean': 0, 'std': 5, 'min_val': -6, 'max_val': 5},
  'cafe_or_restaurant': {'mean': 3, 'std': 7, 'min_val': -5, 'max_val': 10},
  'open-air_market': {'mean': 1, 'std': 7, 'min_val': -5, 'max_val': 10},
  'station_hall': {'mean': 0, 'std': 6, 'min_val': -6, 'max_val': 7},
  'shop': {'mean': 3, 'std': 6, 'min_val': -3, 'max_val': 10},
  'construction_site': {'mean': -5, 'std': 6, 'min_val': -10, 'max_val': 5},
  'bar_nightclub_or_concert_hall': {'mean': -3, 'std': 8, 'min_val': -8, 'max_val': 3},
  'billiard_pool_hall': {'mean': 3, 'std': 6, 'min_val': -3, 'max_val': 10},
  'kid_game_hall': {'mean': 1, 'std': 7, 'min_val': -5, 'max_val': 10},
  'quiet_street': {'mean': 4, 'std': 7, 'min_val': -2, 'max_val': 12},
  'student_hall': {'mean': 3, 'std': 6, 'min_val': -3, 'max_val': 10},
  'airplane_interior': {'mean': 0, 'std': 5, 'min_val': -7, 'max_val': 5},
  'church': {'mean': 6, 'std': 8, 'min_val': -3, 'max_val': 12},
  'office': {'mean': 3, 'std': 7, 'min_val': -5, 'max_val': 10},
  'train_station': {'mean': 1, 'std': 6, 'min_val': -5, 'max_val': 7},
  'street_balcony': {'mean': 3, 'std': 7, 'min_val': -5, 'max_val': 10},
  'home': {'mean': 4, 'std': 7, 'min_val': -2, 'max_val': 12},
  'living_room': {'mean': 3, 'std': 7, 'min_val': -3, 'max_val': 12},
  'library': {'mean': 3, 'std': 8, 'min_val': -5, 'max_val': 15},
  'synthetic_stationary': {'mean': 0, 'std': 8, 'min_val': -10, 'max_val': 10},
}




# Augmentation params
IMPULSE_AUGMENT = True
IMPULSE_AUGMENTATION_PARAMS = [
  SevenBandParametricEQ(min_gain_db=-12, max_gain_db=12, p=0.5),
  PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
  TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5, leave_length_unchanged=False),
  # ApplyImpulseResponse(p=0.5),
]


# rt60 parameters
MIN_RT_60 = 0.2
MAX_RT_60 = 2.0


# possible values: "rir", "ir", "", "mic_ir", "speaker_ir"
IR_TYPES = ['rir', 'ir']


# Background
BACKGROUND_AUGMENT = True

# Final reverb on everything
FINAL_REVERB = True

SCENE_DBA = {
    'car_interior': {'mean': 65, 'std': 5, 'min_val': 50, 'max_val': 75},
    'bus_interior': {'mean': 70, 'std': 6, 'min_val': 60, 'max_val': 80},
    'train_interior': {'mean': 75, 'std': 7, 'min_val': 65, 'max_val': 85},
    'pedestrian_street_or_square': {'mean': 60, 'std': 8, 'min_val': 45, 'max_val': 75},
    'busy_street': {'mean': 75, 'std': 10, 'min_val': 60, 'max_val': 90},
    'airport': {'mean': 70, 'std': 12, 'min_val': 55, 'max_val': 85},
    'park': {'mean': 50, 'std': 5, 'min_val': 40, 'max_val': 60},
    'shopping_mall': {'mean': 65, 'std': 8, 'min_val': 50, 'max_val': 80},
    'metro_station': {'mean': 75, 'std': 10, 'min_val': 60, 'max_val': 90},
    'truck_interior': {'mean': 80, 'std': 7, 'min_val': 70, 'max_val': 90},
    'cafe_or_restaurant': {'mean': 60, 'std': 10, 'min_val': 45, 'max_val': 75},
    'open-air_market': {'mean': 65, 'std': 8, 'min_val': 50, 'max_val': 80},
    'station_hall': {'mean': 65, 'std': 10, 'min_val': 50, 'max_val': 80},
    'shop': {'mean': 55, 'std': 7, 'min_val': 45, 'max_val': 70},
    'construction_site': {'mean': 85, 'std': 10, 'min_val': 70, 'max_val': 100},
    'bar_nightclub_or_concert_hall': {'mean': 80, 'std': 15, 'min_val': 65, 'max_val': 100},
    'billiard_pool_hall': {'mean': 65, 'std': 10, 'min_val': 50, 'max_val': 80},
    'kid_game_hall': {'mean': 70, 'std': 12, 'min_val': 55, 'max_val': 85},
    'quiet_street': {'mean': 50, 'std': 5, 'min_val': 40, 'max_val': 60},
    'student_hall': {'mean': 55, 'std': 8, 'min_val': 45, 'max_val': 70},
    'airplane_interior': {'mean': 80, 'std': 10, 'min_val': 70, 'max_val': 95},
    'church': {'mean': 40, 'std': 5, 'min_val': 30, 'max_val': 50},
    'office': {'mean': 50, 'std': 7, 'min_val': 40, 'max_val': 65},
    'train_station': {'mean': 70, 'std': 10, 'min_val': 55, 'max_val': 85},
    'street_balcony': {'mean': 60, 'std': 8, 'min_val': 45, 'max_val': 75},
    'home': {'mean': 45, 'std': 5, 'min_val': 35, 'max_val': 55},
    'living_room': {'mean': 40, 'std': 5, 'min_val': 30, 'max_val': 50},
    'library': {'mean': 35, 'std': 5, 'min_val': 25, 'max_val': 45},
    'synthetic_stationary': {'mean': 65, 'std': 15, 'min_val': 35, 'max_val': 95},
}
