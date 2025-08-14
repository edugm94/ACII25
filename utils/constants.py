PATH_DE_FEATURES = "/home/xxx/phd/code/DDBB/SEED/SEED_EEG/ExtractedFeatures/"
PATH_DE_FEATURES_SEED4 = "/home/xxx/phd/code/DDBB/SEED_IV/eeg_feature_smooth/"
PATH_DE_FEATURES_SEED5 = "/home/xxx/phd/code/DDBB/SEED-V/EEG_DE_features/"
PATH_DE_FEATURES_SEED7 = "/home/xxx/phd/code/DDBB/SEED-VII/EEG_features/"
PATH_RAW_EEG = "/home/xxx/phd/code/DDBB/SEED/SEED_EEG/Preprocessed_EEG/"

MODEL2SAVE = "Results/"

emotion2num = {
    "seed": {"Sad": 0, "Neutral": 1, "Happy": 2},
    "seed4": {"Neutral": 0, "Sad": 1, "Fear": 2, "Happy": 3},
    "seed5": {"Disgust": 0, "Fear": 1, "Sad": 2, "Neutral": 3, "Happy": 4},
    "seed7": {"Happy": 0, "Surprise": 1, "Neutral": 2, "Disgust": 3, "Fear": 4, "Sad": 5, "Anger": 6}
}

num2emotion = {
    "seed": {0: "Sad", 1: "Neutral", 2: "Happy"},
    "seed4": {0: "Neutral", 1: "Sad", 2: "Fear", 3: "Happy"},
    "seed5": {0: "Disgust", 1: "Fear", 2: "Sad", 3: "Neutral", 4: "Happy"},
    "seed7": {0: "Happy", 1: "Surprise", 2: "Neutral", 3: "Disgust", 4: "Fear", 5: "Sad", 6: "Anger"}
}

pt_list = {
    # "seed": [1],
    "seed": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    "seed4": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    "seed5": [1,2,3,4,5,6,8,9,10,11,12,13,14,15],
    "seed7": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
}

session_list = {
    "seed": [1, 2, 3],
    "seed4": [1, 2, 3],
    "seed5": [1, 2, 3],
    "seed7": [1, 2, 3, 4],
}

# Indexes show the beginning and end of the folds
k_folds_id = {
    "seed": {1: [1, 3], 2:[4, 6], 3:[7, 9], 4:[10, 12], 5:[13, 15]},
    "seed7": {1: [1, 5], 2:[6, 10], 3:[11, 15], 4:[16, 20]},
}