import numpy as np
import tensorflow as tf

tensor_1d = np.array([1.3, 1, 4.0, 23.99])

print "With numpy"

print "Array: %s" % tensor_1d
print "Index 0: %d" % tensor_1d[0]
print "Index 2: %d" % tensor_1d[2]

print "Dimensions: %s" % tensor_1d.ndim
print "Shape: %s" % tensor_1d.shape
print "Datatype: %s" % tensor_1d.dtype

tf_tensor = tf.convert_to_tensor(tensor_1d,dtype=tf.float64)

print "With tensorflow"

with tf.Session() as sess:
    print "Array: %s" % sess.run(tf_tensor)
    print "Index 0: %d" % sess.run(tf_tensor[0])
    print "Index 2: %d" % sess.run(tf_tensor[2])

tensor_2d=np.array([(1,2,3,4),(4,5,6,7),(8,9,10,11),(12,13,14,15)])
print tensor_2d
print tensor_2d[3][3]
print tensor_2d[0:2,0:2]
