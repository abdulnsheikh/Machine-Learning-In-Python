import tensorflow as tf
import numpy as np


totalTrainingSet = 50
kg = np.array([x for x in range(0, totalTrainingSet)], dtype=float)
lb = np.array([x * 2.2 for x in range(0, totalTrainingSet)], dtype=float)

model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, input_shape=[1]))
model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(.1))
model.fit(kg,lb,epochs=500, verbose=False)

print(model.predict([100]))
