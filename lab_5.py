import tensorflow as tf
import numpy as np

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

reader = tf.TextLineReader()
k, v = reader.read('data-03-diabetes.csv')
record_defaults = [[0], [0], [0], [0],[0], [0],[0], [0]]
#xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
#xy = tf.decode_csv('data-03-diabetes.csv', tf.float32, ',')
x1, x2, x3, x4, x5, x6, x7, x8, y = tf.decode_csv(v, record_defaults=record_defaults, field_delim=',')
features = tf.stack([x1, x2, x3, x4, x5, x6, x7, x8])
#x_data = xy[:, 0:-1]
#y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bial')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
            feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
