from tensorflow.keras.preprocessing.image import ImageDataGenerator
#CNN,  MobileNet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#for converting image to array
from tensorflow.keras.preprocessing.image import img_to_array
#for loading the image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
#for digitizing lables
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os



DIRECTORY = r"C:\Mask Detection\CODE\Face-Mask-Detection-master\dataset"
#classification lables
CATEGORIES = ["with_mask", "without_mask"]


#arrays of the images will be added here
data = []

labels = []

#going through both the classes in the categories ie. with mask and then without mask
for category in CATEGORIES:
	#merging the categories into the directory
    path = os.path.join(DIRECTORY, category)
	#going through the list
    for img in os.listdir(path):
	#further adding img in the path as image path
    	img_path = os.path.join(path, img)
	#loading image inside the image path via load_img() in keras.preprocessing.image
    	image = load_img(img_path, target_size=(224, 224))
	#img_to_array() in keras.preprocessing.image
    	image = img_to_array(image)
	#for feeding into MobileNet CNN
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# label needs to be digitized now using Sklearn.preprocessing()
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
#converting into arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
#spliting traing and testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
#LR 
INIT_LR = 1e-4
EPOCHS = 20
#batchsize
BS = 32

# this is for creating variations in images 
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#load the CNN
#imagenet is a pre traind weight
#3 for RGB
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# constructing our fully connected layer
headModel = baseModel.output
#pooling
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
#128 neurons with relu which is good for images
headModel = Dense(128, activation="relu")(headModel)
#avoiding overfitting
headModel = Dropout(0.5)(headModel)
#softamax is good for binary classification
headModel = Dense(2, activation="softmax")(headModel)


#our model will be the combined CNN and FC
model = Model(inputs=baseModel.input, outputs=headModel)

#only traing the FC and not the base model
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# training our model(FC)
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# prediction
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

#highest probabilty label is assigned 
predIdxs = np.argmax(predIdxs, axis=1)

# classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# saving our custom model as mask_detector.model 
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plotting the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
