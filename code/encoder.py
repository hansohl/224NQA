import tensorflow as tf



class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size #hidden state size (l)
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        #read inputs
        question, paragraph = inputs
        q_mask, p_mask = masks

        #run encode LSTM to get representations
        with tf.variable_scope('encoder') as scope:
            encode_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            q_outputs, q_end_state = tf.nn.dynamic_rnn(encode_cell, question, sequence_length=q_mask, dtype=tf.float32)
            #note LSTM returns a pair of hidden states (c, h) in end_state
            p_outputs, p_end_state = tf.nn.dynamic_rnn(encode_cell, paragraph, sequence_length=p_mask, dtype=tf.float32)
            
            #make and concat sentinel values
            #uniform random float init [0,1)
            q_sentinel = tf.get_variable("q_sent", initializer=tf.random_uniform([1, 1, self.size]))
            p_sentinel = tf.get_variable("p_sent", initializer=tf.random_uniform([1, 1, self.size]))
            q_sents = tf.tile(q_sentinel, tf.constant([tf.shape(question)[0], 1, 1]))
            p_sents = tf.tile(p_sentinel, tf.constant([tf.shape(paragraph)[0], 1, 1]))
            q_outputs = tf.concat(1, [q_outputs, q_sents]) #concat along seq axis
            P = tf.concat(1, [p_outputs, p_sents])
            
            #get final Q rep:
            W = tf.get_variable("W_qenc", shape=[self.size, self.size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b_qenc", shape=[tf.shape(question)[0], tf.shape(question)[1], self.size])
            Q = tf.tanh(tf.matmul(W, q_outputs) + b)            
            scope.reuse_variables()
            
        #affinity matrix from batch matmuln and resulting attention weights
        L = tf.batch_matmul(Q, P, adj_y=True)
        A_P = tf.nn.softmax(L)
        A_Q = tf.nn.softmax(tf.transpose(L))
            
        #get attention contexts
        C_Q = tf.batch_matmul(tf.tile(A_Q, tf.constant([tf.shape(question)[0], 1, 1])), P)
        expanded_Q = tf.concat(-1, [Q, C_Q])
        C_D = tf.batch_matmul(tf.tile(A_P, tf.constant([tf.shape(paragraph)[0], 1, 1])), expanded_Q)
        
        #run biLSTM over context C_D
        with tf.variable_scope('encoder_BLSTM') as scope:
            encode2_f_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            encode2_b_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            outputs, end_state = tf.nn.bidirectional_dynamic_rnn(encode2_f_cell, encode2_b_cell, C_D, sequence_length=p_mask, dtype=tf.float32) #init state?
            self.reuse_variables()
            
        #form knowledge rep U from C_D biLSTM, cut off outputs corresponding to sentinel (last in sequence dim)
        self.U = tf.concat(2, outputs)[:,:-1,:]
        return self.U
        
        
        
