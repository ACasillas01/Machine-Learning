import tensorflow as tf
import matplotlib.pyplot as plt

tiempo = tf.constant([5,7,12,16,20], dtype=tf.float32)
masa = tf.constant([40,120,180,210,240], dtype= tf.float32)

def plot_data(x, y):
    plt.scatter(x,y,label="Masa en el tiempo", color='blue')
    plt.xlabel("Tiempo")
    plt.ylabel("Masa")
    plt.title("Masa durante una reacción quimica")
    plt.legend()
    plt.show()

plot_data(tiempo, masa)

def simple_linear_regression(X, w0, b): #Asume data is tensor
    # r = tf.add(tf.multiply(w0, X) , b )
    r = (w0 * X) + b
    return r
r = simple_linear_regression(tiempo, 13, 1)
print(r)

def plot_r(x, y, r):
    plt.scatter(x,y,label="Masa en el tiempo", color='blue')
    plt.scatter(x,r,label="Masa en el tiempo", color='red')
    plt.xlabel("Tiempo")
    plt.ylabel("Masa")
    plt.title("Masa durante una reacción quimica")
    plt.legend()
    plt.show()

plot_r(tiempo, masa, r.numpy())

# 1 / n SUM till n form i=1 of (y - ^y)²

def mse_loss(Y, Yp):
    e = Y - Yp
    se = tf.multiply(e,e)
    mse = tf.reduce_mean(se, axis=0)
    return mse

loss = mse_loss(masa, r)
print(loss)