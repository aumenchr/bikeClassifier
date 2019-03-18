TRAINING_IMAGES_DIR = 'training_images' # Used to locate training images for train.py
TESTING_IMAGES_DIR = 'test_images' # Used to locate testing images for train.py (note, this directory must resemble the training directory, just with different images)
FINAL_TESTING_IMAGES_DIR = 'pure_test' # Used by test.py to display images and predictions, just needs to contain .jpg images without any structure
LABEL_FILENAME = 'labels.txt' # Label names will be written to this file path in the directory train.py is run from
GRAPH_FILENAME = 'graph.pb' # Graph will be written to this file path in the directory train.py is run from
GRAPH_LOC = 'graph' # Intermediate graphs will be written to this directory, if set to '' will be written to a temp directory
FINAL_TENSOR_NAME = 'final_tensor_name' # Naming variable needed to be identical for train.py and test.py
EPOCHS = 60 # Number of training runs
BATCH_SIZE = 50 # Number of random images to train on
IMAGE_SIZE = 64 # Resizes all images to IMAGE_SIZExIMAGE_SIZE
CHANNELS = 3 # Adapts images to/from either 1 or 3 channels, fails if set to any other number
SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)
