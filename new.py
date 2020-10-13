import numpy as np 
#import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
#from keras.models import Sequential 
#from keras.layers import Dense
import matplotlib.pyplot as plt  
from sklearn.model_selection import KFold
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import r2_score
import random
from sklearn import metrics

#importing data
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#seed = 7
#np.random.seed(seed)
#kfold = KFold(n_splits=6, shuffle=None, random_state=False)
kfold = KFold(n_splits=6)

model = keras.Sequential([
keras.layers.Flatten(input_shape=(28,28)),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dense(64, activation="relu"),
keras.layers.Dense(10, activation="softmax")])

i = 1;

for train, test in kfold.split(train_images, train_labels):
	#model = keras.Sequential([
	#keras.layers.Flatten(input_shape=(28,28)),
	#keras.layers.Dense(128, activation="relu"),
	#keras.layers.Dense(10, activation="softmax")])
	print("fold: ", i)
	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	model.fit(train_images[train], train_labels[train], epochs=5, verbose=1)
	i = i+1

test_loss, test_acc = model.evaluate(train_images[test], train_labels[test])
#print(len(train_images[test]))
#print(len(train_images[train]))
print("Tested accuracy of the model: ", test_acc)

#Save epochs
#model.save("model1.h5")
#runModel = keras.models.load_model("model1.h5")
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Tested accuracy of the model: ", test_acc)

var = 9999
#predict the desire output of define var=9999 variable
prediction = model.predict(test_images)

plt.grid(False)
plt.imshow(test_images[var], cmap=plt.cm.binary)
plt.xlabel("Actual: "+ class_names[test_labels[var]])
plt.title("Prediction: "+ class_names[np.argmax(prediction[var])])
plt.show()
