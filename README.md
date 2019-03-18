## CountDice
A Convultional Neural Network (CNN) designed to classify images.  
Developed to classify bikes as either 'Mountain Bikes' or 'Road Bikes'

### Installation
* git clone https://github.com/aumenchr/countDice.git

### Train Network
* cd <path/to>/bikeClassifier
* Modify settings (settings.py) as desired:
  * EPOCHS : Number of training steps; higher = longer train time, lower = shorter.  Suggested 60
  * IMAGE_SIZE : Images are expected to be squared and will be resized to this value for both height and width.  Suggested 64
  * CHANNELS : 1 for grayscale, 3 for RGB.
  * All the strings are used to inform the program where files are kept, comments provide details if your making changes.
* Ensure images exist in <TRAINING_IMAGES_DIR>/<class_name> and <TESTING_IMAGES_DIR>/<class_name>
* ``` python train.py ```

### Classify an Image
* cd <path/to>/bikeClassifier
* KEEP SETTINGS THE SAME AS USED TO TRAIN NETWORK
  *If unsure what the settings were, modify and retrain the model
* Ensure images exist in <FINAL_TESTING_IMAGES_DIR>, do not place in subfolders
* ``` python test.py ```
