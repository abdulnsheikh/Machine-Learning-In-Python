import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_label),(test_images, test_label) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 

test_images = test_images / 255
train_images = train_images /255

model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28)),
	tf.keras.layers.Dense(128,activation='relu'),
	tf.keras.layers.Dense(10)
	])


model.compile(optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])


model.fit(train_images,train_label, epochs=10)


test_loss, test_acc = model.evaluate(test_images, test_label, verbose=2)
print("test_acc - " + str(test_acc))

probability_model = tf.keras.Sequential([model,
									tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(np.argmax(predictions[0]))


plt.figure()
plt.imshow(test_images[0])
plt.xlabel(class_names[np.argmax(predictions[0])])
plt.show()