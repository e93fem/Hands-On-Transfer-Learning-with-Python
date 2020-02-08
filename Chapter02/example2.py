# EXAMPLE of Tensor Operations using tensorflow.
import tensorflow as tf

# Initialize 3 constants: 2 vectors, a scalar and a 2D tensor
x1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])
b = tf.constant(10)
W = tf.constant(-1, shape=[4, 2])

# Elementwise Multiply/subtract
res_elem_wise_mult = tf.multiply(x1, x2)
res_elem_wise_sub = tf.subtract(x1, x2)

# dot product of two tensors of compatable shapes
res_dot_product = tf.tensordot(x1, x2, axes=1)

# broadcasting : add scalar 10 to all elements of the vector
res_broadcast = tf.add(x1, b)

# Calculating Wtx
res_matrix_vector_dot = tf.multiply(tf.transpose(W), x1)

# scalar multiplication
scal_mult_matrix = tf.scalar_mul(scalar=10, x=W)

# Initialize Session and execute
with tf.Session() as sess:
    output = sess.run([res_elem_wise_mult, res_elem_wise_sub, res_dot_product, res_broadcast, res_matrix_vector_dot,
                       scal_mult_matrix])
    print(output)
