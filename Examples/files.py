import pandas as pd
import tensorflow as tf

file_path='./diabetes.csv'
df = pd.read_csv(file_path)

print(df.head(4))

target = df.pop('Outcome')
print(target.head(4))

variables = ['Glucose', 'Insulin', 'Age']
features = df[variables]
print(features.head(4))

tf_features = tf.convert_to_tensor(features, dtype=tf.float32)
print(tf_features.shape)
print(tf_features[:5])

tf_targets = tf.convert_to_tensor(target, dtype=tf.float32)
print(tf_targets.shape)
print(tf_targets[:5])