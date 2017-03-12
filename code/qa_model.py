from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from qa_data import PAD_ID
from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
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
        
        #calculations
        with tf.variable_scope('enc'):
            encode_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            q_outputs, q_end_state = tf.nn.dynamic_rnn(encode_cell, question, sequence_length=masks, dtype=tf.float32) #LSTM returns a pair of hidden states (c, h)
        q_end_state = q_end_state[0]
        q_end_state = tf.expand_dims(q_end_state, axis=1)
        q_states = tf.tile(q_end_state, [1, tf.shape(paragraph)[1], 1])
        para_q_rep = tf.concat(2, [paragraph, q_states])

        return para_q_rep


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
        
        with tf.variable_scope('dec_s'):
            decode_cell_s = tf.nn.rnn_cell.BasicLSTMCell(1) #self.output_size?
            s_outputs, s_end_state = tf.nn.dynamic_rnn(decode_cell_s, knowledge_rep, sequence_length=masks, dtype=tf.float32)
        with tf.variable_scope('dec_e'):
            decode_cell_e = tf.nn.rnn_cell.BasicLSTMCell(1)
            e_outputs, e_end_state = tf.nn.dynamic_rnn(decode_cell_e, knowledge_rep, sequence_length=masks, dtype=tf.float32)
        
        return tf.squeeze(s_outputs), tf.squeeze(e_outputs) #cast to make data scalar?

class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        #extra args
        self.FLAGS  = args[0]
        embed_size = self.FLAGS.embedding_size
        output_size = self.FLAGS.output_size
        

        # ==== set up placeholder tokens ========
        self.q_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="q_place") #batch by seq (None, None)
        self.p_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="p_place")
        self.q_mask_placeholder = tf.placeholder(tf.int32, shape=[None], name="q_mask") #batch (None)
        self.p_mask_placeholder = tf.placeholder(tf.int32, shape=[None], name="p_mask")
        self.s_labels_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="s_place") #batch by output seq
        self.e_labels_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="e_place")

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system(encoder, decoder)
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.optimizer = get_optimizer(self.FLAGS.optimizer)(self.FLAGS.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

    def setup_system(self, encoder, decoder):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        inputs = (self.distr_q, self.distr_p)

        encoding = encoder.encode(inputs, self.q_mask_placeholder, None)
        self.s_ind_probs, self.e_ind_probs = decoder.decode(encoding, self.p_mask_placeholder)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            s_losses = tf.nn.softmax_cross_entropy_with_logits(self.s_ind_probs, self.s_labels_placeholder)
            e_losses = tf.nn.softmax_cross_entropy_with_logits(self.e_ind_probs, self.e_labels_placeholder)
            self.loss = tf.reduce_mean(s_losses) + tf.reduce_mean(e_losses)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embedding_files = np.load("data/squad/glove.trimmed.100.npz")
            self.pretrained_embeddings = tf.constant(embedding_files["glove"], dtype=tf.float32)
            self.distr_q = tf.nn.embedding_lookup(self.pretrained_embeddings, self.q_placeholder)
            self.distr_p = tf.nn.embedding_lookup(self.pretrained_embeddings, self.p_placeholder)
      

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        q, q_lens, p, p_lens = train_x
        s_labels, e_labels = train_y
        input_feed = {}
        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed[self.q_placeholder] = q
        input_feed[self.q_mask_placeholder] = q_lens
        input_feed[self.p_placeholder] = p
        input_feed[self.p_mask_placeholder] = p_lens
        input_feed[self.s_labels_placeholder] = s_labels
        input_feed[self.e_labels_placeholder] = e_labels
        
        output_feed = [self.training_op]

        outputs = session.run(output_feed, input_feed)
        
        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em


    #function to make batch inputs/labels of up to self.FLAGS.batch_size examples
    #output batches are questions, context paras, one hot start labels, one hot end labels
    #pad batches to max batch len (using None for seq-length in the model)
    def make_batch(self, dataset, iteration):
        batch_size = self.FLAGS.batch_size
        train_q, train_p, train_span, val_q, val_p, val_span = dataset
        start_index = iteration*batch_size
        
        #make padded q batch
        qs = train_q[start_index:start_index+batch_size]
        q_seq_lens = np.array([len(q) for q in qs])
        q_seq_mlen = np.max(q_seq_lens)
        q_batch = np.array([q + [PAD_ID]*(q_seq_mlen - len(q)) for q in qs])
        
        #make padded p batch
        ps = train_p[start_index:start_index+batch_size]
        p_seq_lens = np.array([len(p) for p in ps])
        p_seq_mlen = np.max(p_seq_lens)
        p_batch = np.array([p + [PAD_ID]*(p_seq_mlen - len(p)) for p in ps])
        
        #make zero-hot start and end labels
        spans = train_span[start_index:start_index+batch_size]
        starts = np.array([np.eye(1, p_seq_mlen, s_ind) for s_ind, _ in spans])
        starts = np.squeeze(starts)
        ends = np.array([np.eye(1, p_seq_mlen, e_ind) for _, e_ind in spans])
        ends = np.squeeze(ends)

        #update self.output_size for minibatch
        #self.output_size = p_seq_mlen


        return q_batch, q_seq_lens, p_batch, p_seq_lens, starts, ends
        
        
    
    
    

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        
        
        #run main training loop: (only 1 epoch for now)
        train_q, train_p, train_span, val_q, val_p, val_span = dataset
        max_iters = np.ceil(len(train_q)/float(self.FLAGS.batch_size))
        for iteration in range(int(max_iters)):
            print("Current iteration:" + str(iteration))
            q_batch, q_lens, p_batch, p_lens, s_label_batch, e_label_batch = self.make_batch(dataset, iteration)
            lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
            #set annealed lr?
            self.optimize(session, (q_batch, q_lens, p_batch, p_lens), (s_label_batch, e_label_batch))
            
        
        
        
        
        
        
        
        
