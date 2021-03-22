# 机器学习框架搭建
import tensorflow as tf

W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


def inference(X):
    return tf.matmul(X, W) + b


def loss(X, Y):
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
                  [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
                  [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308,
                         220, 311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def train(total_loss):
    learning_rate = 0.0000001
    # return tf.train.AdamOptimizer(0.0000001).minimize(total_loss)
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    print(sess.run(inference([[80., 25.]])))
    print(sess.run(inference([[65., 25.]])))


with tf.Session() as sess:
    X, Y = inputs()

    # init = tf.global_variables_initializer()
    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init = tf.global_variables_initializer()  # 初始化所在的位置至关重要，以本程序为例，使用adam优化器时，会主动创建变量。
    # 因此，如果这时的初始化位置在创建adam优化器之前，则adam中包含的变量会未初始化，然后报错。本行初始化时，可以看到Adam
    # 已经声明，古不会出错
    sess.run(init)
    training_steps = 10000
    saver = tf.train.Saver()  # 模型保存和恢复，当把改行放入for循环中以后，会发现程序执行速度明显变慢
    for step in range(training_steps):
        # saver = tf.train.Saver()
        sess.run(train_op)
        if step % 10 == 0:
            print("loss", sess.run(total_loss))

        # if step % 1000 == 0:
        #     saver.save(sess, r"/Users/terminus/Downloads/my-model", global_step=step)

    evaluate(sess, X, Y)
    # saver.save(sess, r"/Users/terminus/Downloads/my-model", global_step=training_steps)

    coord.request_stop()
    coord.join(threads)
    sess.close()
