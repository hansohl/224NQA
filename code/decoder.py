import tensorflow as tf
import hmn

def LSTMNode(h, c, u, scope, iteration, hidden_size = 200):
    with tf.variable_scope(scope) as scope:
        if iteration > 0:
            scope.reuse_variables()
        Wi = tf.get_variable("Wi", [4 * hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        Ui = tf.get_variable("Ui", [hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        # bi = tf.get_variable("bi", [hidden_size])
        # i = tf.sigmoid(tf.matmul(u, Wi) + tf.matmul(h, Ui) + bi)
        i = tf.sigmoid(tf.matmul(u, Wi) + tf.matmul(h, Ui))

        Wf = tf.get_variable("Wf", [4 * hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        Uf = tf.get_variable("Uf", [hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        # bf = tf.get_variable("bf", initializer=tf.ones([batch_size, hidden_size]), dtype=tf.float32)
        # f = tf.sigmoid(tf.matmul(u, Wf) + tf.matmul(h, Uf) + bf)
        f = tf.sigmoid(tf.matmul(u, Wf) + tf.matmul(h, Uf))

        Wo = tf.get_variable("Wo", [4 * hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        Uo = tf.get_variable("Uo", [hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        # bo = tf.get_variable("bo", [hidden_size])
        # o = tf.sigmoid(tf.matmul(u, Wo) + tf.matmul(h, Uo) + bo)
        o = tf.sigmoid(tf.matmul(u, Wo) + tf.matmul(h, Uo))

        Wc = tf.get_variable("Wc", [4 * hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        Uc = tf.get_variable("Uc", [hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        # bc = tf.get_variable("bc", [hidden_size])
        # c_p = tf.tanh(tf.matmul(u, Wc) + tf.matmul(h, Uc) + bc)
        c_p = tf.tanh(tf.matmul(u, Wc) + tf.matmul(h, Uc))

        ct = f * c + i * c_p
        ht = o * tf.tanh(ct)
        # scope.reuse_variables()

    return ht, ct



class DCNDecoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep, masks, batch_size, iters = 4, hidden_size = 200):
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
        encoding_size = hidden_size * 2
        # with tf.variable_scope('decoder') as scope:
        # extract the size tensors
        #should we get rid of the batch_size param to decoder?
        batch_size = tf.shape(knowledge_rep)[0]
        paragraph_size = tf.shape(knowledge_rep)[1]
        U = knowledge_rep

        hmn_s = "hmn_s"
        hmn_e = "hmn_e"
        lstm_d = "lstm_d"

        # set the initial values
        s = tf.zeros([batch_size], dtype=tf.int32)
        e = tf.fill([batch_size], paragraph_size - 1)

        batch_range = tf.range(batch_size, dtype=tf.int32)
        batch_range = tf.expand_dims(batch_range, 1)

        s_index = tf.concat(1, [batch_range, tf.expand_dims(s, 1)])
        e_index = tf.concat(1, [batch_range, tf.expand_dims(e, 1)])

        # new_u_vec = tf.gather_nd(U, ind)
        u_s = tf.gather_nd(U, s_index)
        u_e = tf.gather_nd(U, e_index)

        h = tf.zeros([batch_size, hidden_size])
        c = tf.zeros([batch_size, hidden_size])

        # iterate and update s and e
        for i in range(iters):
            s, u_s_new, alpha = hmn.HMN(U, h, u_s, u_e, i, hmn_s)
            e, u_e_new, beta = hmn.HMN(U, h, u_s, u_e, i, hmn_e)

            u_s = u_s_new
            u_e = u_e_new

            u_se = tf.concat(1, (u_s, u_e))
            h, c = LSTMNode(h, c, u_se, lstm_d, i, hidden_size)


            # scope.reuse_variables()


        # return tf.squeeze(s), tf.squeeze(e) #cast to make data scalar?
        return alpha, beta
        # return tf.one_hot(s, paragraph_size), tf.one_hot(e, paragraph_size) #cast to make data scalar?
