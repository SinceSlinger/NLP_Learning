

class TextCNN(object):
    """
    A CNN class for sentence classification
    With a embedding layer + a convolutional, max-pooling and softmax layer
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """

        :param sequence_length: The length of our sentences
        :param num_classes:     Number of classes in the output layer(pos and neg)
        :param vocab_size:      The size of our vocabulary
        :param embedding_size:  The dimensionality of our embeddings.
        :param filter_sizes:    The number of words we want our convolutional filters to cover
        :param num_filters:     The number of filters per filter size
        :param l2_reg_lambda:   optional

        这里再注释一下filter_sizes和num_filters。filters_sizes是指filter每次处理几个单词，num_filters是指每个尺寸的处理包含几个filter。

        """
        # set placeholders for variables
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')