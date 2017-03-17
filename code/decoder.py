import tensorflow as tf

def LSTMNode(h, c, u, scope, hidden_size = 200):
    with tf.variable_scope(scope) as scope:
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
        return ht, ct



class Decoder(object):
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

        with tf.variable_scope('decoder') as scope:
            U = knowledge_rep

            hmn_s = "hmn_s"
            hmn_e = "hmn_e"
            lstm_d = "lstm_d"

            s = tf.constant(U[:, 0, :])
            e = tf.constant(U[:, -1, :])
            h = tf.zeros([batch_size, hidden_size])
            c = tf.zeros([batch_size, hidden_size])
            for i in range(iters):
                s_n = HMN(U, h, s, e, hmn_s)
                e_n = HMN(U, h, s, e, hmn_e)

                s = tf.constant(U[:, s_n, :])
                e = tf.constant(U[:, e_n, :])

                u_se = tf.concat(1, (s_n, e_n))
                h, c = LSTMNode(h, c, u_se, lstm_d, hidden_size)


            scope.reuse_variables()


        return tf.squeeze(s_outputs), tf.squeeze(e_outputs) #cast to make data scalar?
