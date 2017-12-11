import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

num_vectors = 1000
num_clusters = 4
num_steps = 100

x_values = []
y_values = []
vector_values = []

for i in xrange(num_vectors):
    if np.random.random() > 0.5:
        x_values.append(np.random.normal(0.4, 0.7))
        y_values.append(np.random.normal(0.2, 0.8))
    else:
        x_values.append(np.random.normal(0.6, 0.4))
        y_values.append(np.random.normal(0.8, 0.5))

vector_values = zip(x_values,y_values)
vectors = tf.constant(vector_values)

#plt.plot(x_values,y_values, 'o', label='Input Data')
#plt.legend()
#plt.show()

n_samples = tf.shape(vector_values)[0]
random_indices = tf.random_shuffle(tf.range(0, n_samples))

begin = [0,]
size = [num_clusters,]
size[0] = num_clusters

centroid_indices = tf.slice(random_indices, begin, size)
centroids = tf.Variable(tf.gather(vector_values, centroid_indices))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

vectors_subtration = tf.subtract(expanded_vectors,expanded_centroids)

euclidean_distances = tf.reduce_sum(tf.square(vectors_subtration), 2)
assignments = tf.to_int32(tf.argmin(euclidean_distances, 0))

partitions = tf.dynamic_partition(vectors, assignments, num_clusters)

update_centroids = tf.concat(0, \
    [tf.expand_dims\
    (tf.reduce_mean(partition, 0), 0)\
    for partition in partitions])

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
