import tensorflow as tf


def evaluate_class(classifier, data):
    acc = 0
    count = 0

    for id, ([I1, I2, D1, D2], _) in enumerate(data):
        P1 = tf.argmax(classifier.call(I1), axis=-1)
        P2 = tf.argmax(classifier.call(I2), axis=-1)

        acc += tf.reduce_sum(tf.where(P1 == tf.cast(D1, dtype=tf.int64), 1, 0))
        acc += tf.reduce_sum(tf.where(P2 == tf.cast(D2, dtype=tf.int64), 1, 0))

        count += I1.shape[0]
        count += I2.shape[0]
    return float(acc / count)


def evaluate_complete(graph, data):
    acc = 0
    counter = 0
    for x, y in data:
        I1, I2, R = x
        for id, ri in enumerate(R):
            x1 = tf.stack([I1[id]] * 19)
            x2 = tf.stack([I2[id]] * 19)
            r = tf.stack([i - 9 for i in range(19)])
            AMC = graph.call([tf.constant([0., 1.]), x1, x2, r])
            int_AMC = tf.reduce_mean(AMC, axis=-1)
            pred = tf.argmax(int_AMC)
            acc += (int(pred - 9) == int(R[id]))
            counter += 1
    return acc / counter

def evaluate_complete_joint(graph, data):
    acc = 0
    counter = 0
    for x, y in data:
        I, R = x
        for id, ri in enumerate(R):
            x1 = tf.stack([I[id]] * 19)
            r = tf.stack([i - 9 for i in range(19)])
            AMC = graph.call([tf.constant([0., 1.]), x1, r])
            int_AMC = tf.reduce_mean(AMC, axis=-1)
            pred = tf.argmax(int_AMC)
            acc += (int(pred - 9) == int(R[id]))
            counter += 1
    return acc / counter
