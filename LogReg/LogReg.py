import tensorflow as tf
import numpy as np

# Generate some synthetic data for demonstration
np.random.seed(0)
X_train = np.random.randn(100, 2)
y_train = (np.sum(X_train, axis=1) > 0).astype(np.float32).reshape(-1, 1)

# Define the logistic regression model
class LogisticRegressionModel:
    def __init__(self, input_dim):
        self.W = tf.Variable(tf.random.normal([input_dim, 1]), name='weight')
        self.b = tf.Variable(tf.zeros([1]), name='bias')

    def __call__(self, X):
        z = tf.matmul(X, self.W) + self.b
        Y = 1/(1 + tf.exp(-z))
        return Y #tf.sigmoid(logits)

# Define the loss function
def loss_fn(model, X, y):
    y_pred = model(X)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y))

# Define the training step
def train_step(model, X, y, learning_rate=0.01):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, X, y)
    gradients = tape.gradient(loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * gradients[0])
    model.b.assign_sub(learning_rate * gradients[1])
    return loss

# Training the model
model = LogisticRegressionModel(input_dim=2)
epochs = 1000
for epoch in range(epochs):
    loss = train_step(model, X_train, y_train)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Testing the model
X_test = np.array([[1.0, 2.0], [-1.0, -2.0]])
y_test_pred = model(X_test)
print("Predictions:", y_test_pred.numpy())