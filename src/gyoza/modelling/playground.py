import tensorflow as tf

class MyLoss(tf.keras.losses.Loss):
    def __init__(self, *args):
        super(MyLoss, self).__init__(*args)

    def call(self, y_true, y_pred):
        print(y_true)
        print(y_pred)
        return tf.constant(0)

if __name__ == "__main__":
    loss = MyLoss()
    loss([tf.ones([3,4]),tf.ones([3,4])], tf.ones([3,4]))