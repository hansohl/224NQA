import tensorflow as tf
import numpy as np


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size #hidden state size (l)
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, keep_prob, encoder_state_input):
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
            with tf.variable_scope('encoder_read_LSTM') as scope:
                encode_cell = tf.nn.rnn_cell.LSTMCell(self.size, initializer=tf.contrib.layers.xavier_initializer())
                encode_cell_d = tf.nn.rnn_cell.DropoutWrapper(encode_cell, keep_prob, keep_prob)
                q_outputs, q_end_state = tf.nn.dynamic_rnn(encode_cell_d, question, sequence_length=q_mask, dtype=tf.float32)
                #note LSTM returns a pair of hidden states (c, h) in end_state
                scope.reuse_variables()
                p_outputs, p_end_state = tf.nn.dynamic_rnn(encode_cell, paragraph, sequence_length=p_mask, dtype=tf.float32)
            
            #make and concat sentinel values
            #uniform random float init [0,1)
            q_sentinel = tf.get_variable("q_sent", initializer=tf.random_uniform([1, 1, self.size]))
            p_sentinel = tf.get_variable("p_sent", initializer=tf.random_uniform([1, 1, self.size]))
            q_sents = tf.tile(q_sentinel, [tf.shape(question)[0], 1, 1])
            p_sents = tf.tile(p_sentinel, [tf.shape(paragraph)[0], 1, 1])
            Qprime = tf.concat(1, [q_outputs, q_sents]) #concat along seq axis
            P = tf.concat(1, [p_outputs, p_sents]) #batch by p by hidden
            
            #get final Q rep:
            #wizardry and voodoo to do einsum 'aij,jk->aik' (batch by seq by hidden matrix multiplied with hidden by hidden)
            W = tf.get_variable("W_qenc", shape=[self.size, self.size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b_qenc", initializer=tf.zeros([self.size]), dtype=tf.float32)
            Qprime_b = tf.reshape(Qprime, [-1, self.size])
            Q = tf.reshape(tf.tanh(tf.matmul(Qprime_b, W) + b), [-1, tf.shape(Qprime)[1], self.size]) #batch by q by hidden
            scope.reuse_variables()
            
        #affinity matrix from batch matmul and resulting attention weights
        Lp = tf.batch_matmul(Q, P, adj_y=True) #batch by q by p
        
        #exp mask over L :(        
        def make_slice_mask(inputs):
            lp, q_len, p_len = inputs
            q_max = tf.shape(lp)[0]
            p_max = tf.shape(lp)[1]
            mask_core = tf.ones([q_len, p_len], dtype=tf.float32) #ones_like?
            mask = tf.pad(mask_core, [[0, q_max-q_len], [0, p_max-p_len]]) #pad out zeros part
            slice_mask = (1.-mask)*1e-30
            return slice_mask
        
        exp_mask = tf.map_fn(make_slice_mask, [Lp, q_mask, p_mask], dtype=tf.float32)
        L = Lp + exp_mask
        
        A_P = tf.nn.softmax(L) #batch by q by p, normalized across p
        A_Q = tf.nn.softmax(L, dim=1) #batch by q by p, normalized across q
            
        #get attention contexts
        C_Q = tf.batch_matmul(A_Q, P)
        expanded_Q = tf.concat(2, [Q, C_Q]) #batch by q by 2*hidden
        C_P = tf.batch_matmul(A_P, expanded_Q, adj_x=True) #batch by p by 2*hidden
        C_P.set_shape([None, None, 2*self.size]) #?
        
        #run biLSTM over context C_P
        BLSTM_input = tf.concat(2, [P, C_P])
        with tf.variable_scope('encoder_BLSTM') as scope:
            with tf.variable_scope('encoder_BLSTM_F') as scope:
                encode2_f_cell = tf.nn.rnn_cell.LSTMCell(self.size, initializer=tf.contrib.layers.xavier_initializer())
                encode2_f_cell_d = tf.nn.rnn_cell.DropoutWrapper(encode2_f_cell, keep_prob, 1.) #no output dropout since decoder drops out at start already
                scope.reuse_variables()
            with tf.variable_scope('encoder_BLSTM_B') as scope:
                encode2_b_cell = tf.nn.rnn_cell.LSTMCell(self.size, initializer=tf.contrib.layers.xavier_initializer())
                encode2_b_cell_d = tf.nn.rnn_cell.DropoutWrapper(encode2_b_cell, keep_prob, 1.)
                scope.reuse_variables()
            
            outputs, end_state = tf.nn.bidirectional_dynamic_rnn(encode2_f_cell_d, encode2_b_cell_d, BLSTM_input, sequence_length=p_mask, dtype=tf.float32) #init state?
            scope.reuse_variables()
            
        #form knowledge rep U from biLSTM, cut off outputs corresponding to sentinel (last in sequence dim)
        U_ext = tf.concat(2, outputs)
        self.U = tf.slice(U_ext, [0,0,0], [-1, tf.shape(U_ext)[1]-1, -1])
        self.U.set_shape([None, None, 2*self.size])
        return self.U
        
        
        
