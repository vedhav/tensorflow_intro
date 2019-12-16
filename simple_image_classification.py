import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

print(train_images[7])

plt.imshow(train_images[7], cmap = plt.cm.binary)
plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28, 28)),
	keras.layers.Dense(128, activation = "relu"),
	keras.layers.Dense(10, activation = "softmax")
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc: ", test_acc)


model.save("image_classification_model.model")

new_model = tf.keras.models.load_model("image_classification_model.model")

predictions = new_model.predict([test_images])

class_names[np.argmax(predictions[242])]

plt.imshow(test_images[242], cmap = plt.cm.binary)
plt.show()
