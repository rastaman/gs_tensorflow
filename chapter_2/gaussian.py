import matplotlib.pyplot as plt
import tensorflow as tf

norm = tf.random_normal([100], mean=0, stddev=2)
with tf.Session() as session:
    plt.hist(norm.eval(),normed=True)
    plt.show()
