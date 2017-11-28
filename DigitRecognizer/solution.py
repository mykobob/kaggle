import tensorflow as tf
from numpy import genfromtxt
import numpy as np

# Load the data we are giving you
def load(filename, label, W=28, H=28):
    data = genfromtxt(filename, delimiter=',').reshape((-1, W*H+label))
    images, labels = data[:,1:].reshape((-1,H,W,1)), data[:,0]
    return images, labels

image_data, label_data = load('train.csv', 1)

print('Input shape: ' + str(image_data.shape))
print('Labels shape: ' + str(label_data.shape))

num_classes = 10

### Tensorflow graph STARTS ### 
inputs = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
inputs = (inputs - 100.) / 72.
conv = tf.contrib.layers.conv2d(inputs=inputs, num_outputs=10, kernel_size=[3,3], stride=2, padding='same', scope='conv1')
conv = tf.contrib.layers.conv2d(inputs=conv, num_outputs=20, kernel_size=[3,3], stride=2, padding='same', scope='conv2')
conv = tf.contrib.layers.conv2d(inputs=conv, num_outputs=40, kernel_size=[3,3], stride=2, padding='same', scope='conv3')

output = tf.layers.dense(tf.contrib.layers.flatten(conv), 10, name='output_layer')
### Tensorflow graph FINISHED ###

print( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]), '/', 100000 )

labels = tf.placeholder(tf.int64, (None), name='labels')
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels))

minimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
correct = tf.equal(tf.argmax(output, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


### TRAINING CODE ###
BS = 32
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

def train(train_func):
    # An epoch is a single pass over the training data
    for epoch in range(20):
        # Let's shuffle the data every epoch
        np.random.seed(epoch)
        np.random.shuffle(image_data)
        np.random.seed(epoch)
        np.random.shuffle(label_data)
        # Go through the entire dataset once
        accs, losss = [], []
        for i in range(0, image_data.shape[0]-BS+1, BS):
            # Train a single batch
            batch_images, batch_labels = image_data[i:i+BS], label_data[i:i+BS]
            acc, loss = train_func(batch_images, batch_labels)
            accs.append(acc)
            losss.append(loss)
        print('[%3d] Accuracy: %0.3f  \t  Loss: %0.3f'%(epoch, np.mean(accs), np.mean(losss)))

def train_func(image, label):
    accuracy_res, loss_res, _ = sess.run([accuracy, loss, minimizer], feed_dict={inputs: image, labels: label})
    return accuracy_res, loss_res

train(train_func)
