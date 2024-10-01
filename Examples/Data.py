# Alejamiento de datos y operaciones en CPU / GPU

import tensorflow as tf

print(tf.config.list_physical_devices())

with tf.device('CPU:0'):
    a = tf.constant(1)
    b = tf.constant(2)
    c = a + b
    print(c)