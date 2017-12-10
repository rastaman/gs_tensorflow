import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import tensorflow as tf

filename = "test.jpg"
input_image = mp_image.imread(filename)

print 'input dim = {}'.format(input_image.ndim)
print 'input shape = {}'.format(input_image.shape)

#plt.imshow(input_image)
#plt.show()

#my_image = tf.placeholder("uint8",[None,None,3])
#slice = tf.slice(my_image,[10,0,0],[160,-1,-1])

#with tf.Session() as session:
#    result = session.run(slice,feed_dict={my_image: input_image})
#    print(result.shape)

#plt.imshow(result)
#plt.show()

x = tf.Variable(input_image,name='x')
model = tf.global_variables_initializer()
with tf.Session() as session:
    x = tf.transpose(x, perm=[1,0,2])
    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()
