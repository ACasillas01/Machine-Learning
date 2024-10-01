import tensorflow as tf
import numpy as np

#Ejemplos de suma de 2 nums
nums = np.array([[1,1],[2,1],[3,1],[4,1],[5,1]])
sum = nums[:,0] + nums[:,1]

# Red Neuronal recibe 2 nums y produce 1
neural_network = tf.keras.Sequential([tf.keras.Input((2,)),
                                     tf.keras.layers.Dense(5),
                                     tf.keras.layers.Dense(3),
                                     tf.keras.layers.Dense(1)
                                     ])

neural_network.compile(loss='mse', optimizer='adam')

#Entrenamiento
neural_network.fit(nums, sum, epochs=1000, verbose=0)

#Aglgunas sums de 2 nums
num1 = float(input("Ingresa un numero: "))
num2 = float(input("Ingresa otro numero: "))

new_nums = np.array([[num1, num2]])
predictions = neural_network.predict(new_nums)
print(predictions)