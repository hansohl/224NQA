import tensorflow as tf
import numpy as np


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
            with tf.variable_scope('encoder_read_LSTM') as scope:
                encode_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
                q_outputs, q_end_state = tf.nn.dynamic_rnn(encode_cell, question, sequence_length=q_mask, dtype=tf.float32)
                #note LSTM returns a pair of hidden states (c, h) in end_state
                scope.reuse_variables()
                p_outputs, p_end_state = tf.nn.dynamic_rnn(encode_cell, paragraph, sequence_length=p_mask, dtype=tf.float32)
            
            #make and concat sentinel values
            #uniform random float init [0,1)
            q_sentinel = tf.get_variable("q_sent", initializer=tf.random_uniform([1, 1, self.size]))
            p_sentinel = tf.get_variable("p_sent", initializer=tf.random_uniform([1, 1, self.size]))
            q_sents = tf.tile(q_sentinel, [tf.shape(question)[0], 1, 1])
            p_sents = tf.tile(p_sentinel, [tf.shape(paragraph)[0], 1, 1])
            q_outputs = tf.concat(1, [q_outputs, q_sents]) #concat along seq axis
            P = tf.concat(1, [p_outputs, p_sents])
            
            #get final Q rep:
            #wizardry and voodoo to do einsum 'aij,jk->aik' (batch by seq by hidden matrix multiplied with hidden by hidden)
            W = tf.get_variable("W_qenc", shape=[self.size, self.size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b_qenc", shape=[self.size], dtype=tf.float32, validate_shape=False)
            W_batch = tf.tile(tf.expand_dims(W,0), [tf.shape(question)[0], 1, 1])
            Q = tf.tanh(tf.batch_matmul(q_outputs, W_batch) + b)
            #Q = tf.tanh(tf.einsum('aij,jk->aik', q_outputs, W) + b)
            scope.reuse_variables()
            
        #affinity matrix from batch matmuln and resulting attention weights
        L = tf.batch_matmul(Q, P, adj_y=True)
        print(L.get_shape())
        A_P = tf.nn.softmax(L)
        print(A_P.get_shape())
        A_Q = tf.nn.softmax(tf.transpose(L))
        print(A_Q.get_shape())
            
        #get attention contexts
        C_Q = tf.batch_matmul(A_Q, P)
        expanded_Q = tf.concat(2, [Q, C_Q])
        C_D = tf.batch_matmul(A_P, expanded_Q)
        
        #run biLSTM over context C_D
        with tf.variable_scope('encoder_BLSTM') as scope:
            with tf.variable_scope('encoder_BLSTM_F') as scope:
                encode2_f_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
                scope.reuse_variables()
            with tf.variable_scope('encoder_BLSTM_B') as scope:
                encode2_b_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
                scope.reuse_variables()
            C_D.set_shape([None, None, 2*self.size])
            outputs, end_state = tf.nn.bidirectional_dynamic_rnn(encode2_f_cell, encode2_b_cell, C_D, sequence_length=p_mask, dtype=tf.float32) #init state?
            scope.reuse_variables()
        #form knowledge rep U from C_D biLSTM, cut off outputs corresponding to sentinel (last in sequence dim)
        self.U = tf.concat(2, outputs)[:,:-1,:]
        return self.U
        
        
        
