import tensorflow as tf
import numpy as np

print('===========================================')
#===================== Constant Tensors ==========================

# ==================== 1 Dimension =============================
tensor00 = tf.constant(4) #, dtype =tf.float32
print(tensor00)
print(tensor00.shape)
print(tensor00.numpy())

vector = tf.constant([1, 2, 3, 4, 5, 6]) #Vector
print(vector)
print(vector.shape)
print(vector.numpy())
print(vector[2].numpy())

# ===================== 2 Dimensions ===========================
a = np.array([[1, 2, 3], [4, 5, 6]])
tensor02 = tf.constant(a)

matrix = tf.constant([[1,2], [3,4]])
print('========== Matrix ==========')
print(matrix.shape)
print(matrix.numpy())

print('=========== tensor03 ===========')
tensor03 = tf.constant([1, 2], dtype=tf.float64)
tensor04 = tf.constant(1, shape=(2, 3, 4))
print(tensor04.shape)
print(tensor04.numpy())
tensor05 = tf.constant([1, 2, 3, 4, 5, 6],
shape=[3, 2])

print('===========================================')
#===================== Variable Tensors ==========================
tensor = tf.constant([1, 2, 3, 4, 5, 6])
var01 = tf.Variable(tensor, name='var01')
print(var01)

var02 = tf.Variable([5, 6], dtype=tf.int32)
var02.assign([1, 2])
print(var02.numpy())
var02.assign_add([2, 3])
print(var02.numpy())
var02.assign_sub([1, 1])
print(var02.numpy())

print('==============================================')
#================== Operations with Tensors =====================
#================== Basic Operations ===========================

#================== Addition ===========================
#================== Scalar + Scalar ================
t1= tf.constant(10)
t2 = tf.constant(5)
ts = tf.add(t1, t2)
print(ts)

#================ Scalar + Vector ===================
t1 = tf.constant([1, 2, 3])
t2 = tf.constant(5)
print(tf.add(t1, t2))

#=============== Multiplication =====================
scalar1 = tf.constant(2, dtype=tf.float32)
scalar2 = tf.constant(2, dtype=tf.float32)
r = tf.multiply(scalar1, scalar2)
print(r)

vector01 = tf.constant([1,2,3,4], dtype=tf.float32)
vector02 = tf.constant([1,2,3,4], dtype=tf.float32)
r = tf.multiply(vector01, vector02)
print(r)

vector01 = tf.constant([1,2,3,4], dtype=tf.float32)
vector02 = tf.constant([1,2,3,4], dtype=tf.float32)
r = vector01 * vector02
print(r)

A = tf.constant([[1,2]])
B = tf.constant([[3],[4]])
print(A)
print(B)

r = tf.matmul(A,B)
print(r)