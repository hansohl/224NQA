import tensorflow as tf


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep, masks):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        with tf.variable_scope('dec_s') as scope:
            decode_cell_s = tf.nn.rnn_cell.LSTMCell(200, initializer=tf.contrib.layers.xavier_initializer()) #self.output_size?
            s_outputs, s_end_state = tf.nn.dynamic_rnn(decode_cell_s, knowledge_rep, sequence_length=masks, dtype=tf.float32)
            scope.reuse_variables()
        with tf.variable_scope('dec_e') as scope:
            decode_cell_e = tf.nn.rnn_cell.LSTMCell(200, initializer=tf.contrib.layers.xavier_initializer())
            e_outputs, e_end_state = tf.nn.dynamic_rnn(decode_cell_e, knowledge_rep, sequence_length=masks, dtype=tf.float32)
            scope.reuse_variables()
        with tf.variable_scope('dec_s_fc') as scope:
            Ws = tf.get_variable("W_sdec", shape=[200,1], initializer=tf.contrib.layers.xavier_initializer())
            bs = tf.get_variable("b_sdec", initializer=tf.zeros([1]))
            scope.reuse_variables()
        with tf.variable_scope('dec_e_fc') as scope:
            We = tf.get_variable("W_edec", shape=[200,1], initializer=tf.contrib.layers.xavier_initializer())
            be = tf.get_variable("b_edec", initializer=tf.zeros([1]), dtype=tf.float32)
            scope.reuse_variables()    
        
        s_outputs_b = tf.reshape(s_outputs, [-1, 200])
        s_preds = tf.reshape(tf.matmul(s_outputs_b, Ws)+bs, [-1, tf.shape(s_outputs)[1]])
        e_outputs_b = tf.reshape(e_outputs, [-1, 200])
        e_preds = tf.reshape(tf.matmul(e_outputs_b, We)+be, [-1, tf.shape(e_outputs)[1]])

        #return s_outputs[:,:,0], e_outputs[:,:,0]
        return s_preds, e_preds
