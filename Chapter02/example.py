import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Initialize Gradient Descent at x,y =(2, 2)
x = tf.Variable(2, name='x', dtype=tf.float32)
y = tf.Variable(2, name='y', dtype=tf.float32)
temperature = 50 - tf.square(y) - 2 * tf.square(x)

# Initialize Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)  # 0.1 is the learning rate
train = optimizer.minimize(temperature)
grad = tf.gradients(temperature, [x, y])  # Lets calculate the gradient vector
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    print("Starting at coordinate x={}, y={} and temperature there is {}".format(
        session.run(x), session.run(y), session.run(temperature)))
    grad_norms = []
    for step in range(10):
        session.run(train)
        g = session.run(grad)
        print("step ({}) x={},y={}, T={}, Gradient={}".format(step, session.run(x), session.run(y),
                                                              session.run(temperature), g))
        grad_norms.append(np.linalg.norm(g))

plt.plot(grad_norms)
plt.show()
