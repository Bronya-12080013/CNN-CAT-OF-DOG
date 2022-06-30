import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse

image_size = 64
num_channels = 3
images = []

path = 'cat2.jpg'
image = cv2.imread(path)

image = cv2.resize(image, (image_size, image_size))
images.append(image)
# images = np.array(images,dtype=np.uint8)
images = np.array(images)
images = images.astype('float32')
images = np.multiply(images, 1.0 / 255.0)

# tensorflow 就要四维的
x_batch = images.reshape(1, image_size, image_size, num_channels)


sess = tf.compat.v1.Session()

tf.compat.v1.disable_eager_execution()
saver = tf.compat.v1.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-4950.meta')

saver.restore(sess, './dogs-cats-model/dog-cat.ckpt-4950')
graph = tf.compat.v1.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")

x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 2))

feed_dict_testing = {x: x_batch, y_true: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)

res_label = ['dog', 'cat']
print(res_label[result.argmax()])
