import tensorflow as tf
from numpy import genfromtxt
import numpy as np

# Load the data we are giving you
def load(filename, label, W=28, H=28):
    data = genfromtxt(filename, delimiter=',').reshape((-1, W*H+label))
    if label == 1:
        images, labels = data[:,1:].reshape((-1,H,W,1)), data[:,0]
        return images, labels
    else:
        images = data.reshape((-1,H,W,1))
        return images

image_data, label_data = load('train.csv', 1)
image_val = load('test.csv', 0)

print('Input shape: ' + str(image_data.shape))
print('Labels shape: ' + str(label_data.shape))

# Create a new log directory (if you run low on disk space you can either disable this or delete old logs)
# run: `tensorboard --logdir log` to see all the nice summaries
for n_model in range(1000):
    LOG_DIR = 'log/model_%d'%n_model
    from os import path
    if not path.exists(LOG_DIR):
        break

### Tensorflow graph STARTS ### 
inputs = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
conv = inputs
init_outputs = 30

for i in range(3):
    conv = tf.contrib.layers.conv2d(inputs=conv, num_outputs=init_outputs * int(1.5**i), kernel_size=[3,3], stride=2, padding='same', scope='conv{}'.format(i))

output = tf.layers.dense(tf.contrib.layers.flatten(conv), 10, name='output_layer')
### Tensorflow graph FINISHED ###

print( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]), '/', 100000 )

labels = tf.placeholder(tf.int64, (None), name='labels')
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels))

minimizer = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(loss)
correct = tf.equal(tf.argmax(output, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Let's define some summaries for tensorboard
#COLORS = np.array([(0,0,0), (255,0,0), (0,255,0), (255,255,0), (0,0,255), (255,255,255)], dtype=np.uint8)
#TF_COLORS = tf.constant(COLORS)
#colored_label = tf.gather_nd(TF_COLORS, labels[:,:,:,None])
#colored_output = tf.gather_nd(TF_COLORS, output[:,:,:,None])
#tf.summary.scalar('loss', tf.placeholder(tf.float32, name='loss'))
#tf.summary.scalar('accuracy', tf.placeholder(tf.float32, name='accuracy'))

merged_summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())

### TRAINING CODE ###
BS = 32
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

def train(train_func):
    # An epoch is a single pass over the training data
    for epoch in range(30):
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

### Validation time ### 
with open('mnist_output.txt', 'w') as f:
    f.write('ImageId,Label\n')
    output_vals = sess.run(output, feed_dict={inputs: image_val})
    for i in range(len(output_vals)):
        output_val = output_vals[i]
        idx = np.argmax(output_val)
        f.write('{},{}\n'.format(i + 1, idx))
