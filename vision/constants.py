# Constants

# CNN input image size
INPUT_WIDTH = 30
INPUT_HEIGHT = 60

# Version
version = "4"

# Paths
LETTER_DATA_PATH = "/home/fizzer/enph353/vision/data/no_blur_letters"
LETTER_MODEL_PATH = "/home/fizzer/enph353/vision/training/letter" + version + ".h5"
LETTER_WEIGHTS_PATH = "/home/fizzer/enph353/vision/training/letter" + version + ".ckpt"

NUMBER_DATA_PATH = "/home/fizzer/enph353/vision/data/no_blur_numbers"
NUMBER_MODEL_PATH = "/home/fizzer/enph353/vision/training/number" + version + ".h5"
NUMBER_WEIGHTS_PATH = "/home/fizzer/enph353/vision/training/number" + version + ".ckpt"

BINARY_DATA_PATH = "/home/fizzer/enph353/vision/data/binary"
BINARY_MODEL_PATH = "/home/fizzer/enph353/vision/training/binary" + version + ".h5"
BINARY_WEIGHTS_PATH = "/home/fizzer/enph353/vision/training/binary" + version + ".ckpt"


# Params
MIN_IMAGE_SIZE = 8000
