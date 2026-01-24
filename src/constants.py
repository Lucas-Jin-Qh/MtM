# pyre-strict

SEED = 42

PAD_VALUE = -1

MODALITY_INFO = {
    "spikes": {
        "id": 0,
    },
    "video": {
        "id": 1,
    },
    "choice": {
        "id": 2,
        "dim": 2,
    },
    "block": {
        "id": 3,
        "dim": 3,
    },
    "wheel": {
        "id": 4,
        "dim": 1,
    },
    "left-whisker": {
        "id": 5,
        "dim": 1,
    },
    "right-whisker": {
        "id": 6,
        "dim": 1,
    },
}

# TODO: Add encoding and decoding
MASK_MODE = [
    "temporal", "co_smooth", "intra-region", "inter-region"
]

DISCRETE_BEHAVIOR = ["block", "choice"]

CONTINUOUS_BEHAVIOR = ["wheel", "left-whisker", "right-whisker"]