
from constants import *

from classifier import Classifier

## Plate reading CNNs
letterClassifier = Classifier("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
letterClassifier.loadWeights(LETTER_WEIGHTS_PATH)
letterClassifier.train(LETTER_DATA_PATH, LETTER_WEIGHTS_PATH)

numberClassifier = Classifier("0123456789")
numberClassifier.loadWeights(NUMBER_WEIGHTS_PATH)
numberClassifier.train(NUMBER_DATA_PATH, NUMBER_WEIGHTS_PATH)

binaryClassifier = Classifier("01")
#binaryClassifier.loadWeights(BINARY_WEIGHTS_PATH)
binaryClassifier.train(BINARY_DATA_PATH, BINARY_WEIGHTS_PATH)


letterClassifier.saveLayers(LETTER_MODEL_PATH)
numberClassifier.saveLayers(NUMBER_MODEL_PATH)
binaryClassifier.saveLayers(BINARY_MODEL_PATH)


#letterClassifier.showMetrics()
#numberClassifier.showMetrics()
#binaryClassifier.showMetrics()
