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

'''
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
        q_mask, p_mask = masks

        #run biLSTM over question
        with tf.variable_scope('enc_q') as scope:
            encode_q_f_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            encode_q_b_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            q_outputs, q_end_state = tf.nn.bidirectional_dynamic_rnn(encode_q_f_cell, encode_q_b_cell, question, sequence_length=q_mask, dtype=tf.float32) #LSTM returns a pair of hidden states (c, h)
            scope.reuse_variables()

        #concat end states to get question representation
        q_fwd_state, q_bkwd_state = q_end_state
        self.q_rep = tf.concat(1, (q_fwd_state[0], q_bkwd_state[0])) #q rep is Batch by 2*H_size

        #run biLSTM over paragraph
        with tf.variable_scope('enc_p') as scope:
            encode_p_f_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            encode_p_b_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            p_outputs, p_end_state = tf.nn.bidirectional_dynamic_rnn(encode_p_f_cell, encode_p_b_cell, paragraph, sequence_length=p_mask, dtype=tf.float32) #condition on q rep?
            scope.reuse_variables()
        self.p_rep = tf.concat(2, p_outputs) #concat fwd and bkwd outputs


        #calc scores between paragraph hidden states and q-rep
        self.attention_weights = tf.get_variable("attent_weights", shape=[2*self.size, 2*self.size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        q_attention = tf.matmul(self.q_rep, self.attention_weights)
        unnorm_attention = tf.batch_matmul(self.p_rep, tf.expand_dims(q_attention, axis=-1)) #dims are batch by seq by 1
        self.attention = unnorm_attention/tf.sqrt(tf.reduce_sum(tf.square(unnorm_attention), axis=1, keep_dims=True))
        self.knowledge_rep = tf.multiply(self.p_rep, self.attention)

        return self.knowledge_rep, self.attention


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
            decode_cell_s = tf.nn.rnn_cell.BasicLSTMCell(1) #self.output_size?
            s_outputs, s_end_state = tf.nn.dynamic_rnn(decode_cell_s, knowledge_rep, sequence_length=masks, dtype=tf.float32)
            scope.reuse_variables()
        with tf.variable_scope('dec_e') as scope:
            decode_cell_e = tf.nn.rnn_cell.BasicLSTMCell(1)
            e_outputs, e_end_state = tf.nn.dynamic_rnn(decode_cell_e, knowledge_rep, sequence_length=masks, dtype=tf.float32)
            scope.reuse_variables()

        return tf.squeeze(s_outputs), tf.squeeze(e_outputs) #cast to make data scalar?
'''

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

        #custom gradient handling - can add gradient clipping later
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        grads = [grad for grad, _ in grads_and_vars]
        #if self.config.clip_gradients:
        #    grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        #    grads_and_vars = [(grads[i], grads_and_vars[i][1]) for i in range(len(grads_and_vars))]
        self.grad_norm = tf.global_norm(grads)
        self.training_op = self.optimizer.apply_gradients(grads_and_vars)

        #default (boring!) trainingop
        #self.training_op = self.optimizer.minimize(self.loss)



    def setup_system(self, encoder, decoder):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        inputs = (self.distr_q, self.distr_p)

        encoding = encoder.encode(inputs, (self.q_mask_placeholder, self.p_mask_placeholder), None)
        self.s_ind_probs, self.e_ind_probs = decoder.decode(encoding, self.p_mask_placeholder, self.FLAGS.batch_size)

        self.attention = attention #store attention vector for analysis


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
        :return: training_op eval, loss
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

        #set the quantities we track/return during training
        output_feed = [self.training_op, self.loss, self.s_ind_probs, self.e_ind_probs, self.e_labels_placeholder, self.p_mask_placeholder, self.grad_norm, self.attention]

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
        q, q_lens, p, p_lens = test_x
        input_feed = {}
        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        input_feed[self.q_placeholder] = q
        input_feed[self.q_mask_placeholder] = q_lens
        input_feed[self.p_placeholder] = p
        input_feed[self.p_mask_placeholder] = p_lens
        # we don't have the ground truth to loook at, so we dont pass in labels
        # input_feed[self.s_labels_placeholder] = s_labels
        # input_feed[self.e_labels_placeholder] = e_labels

        output_feed = [self.s_ind_probs, self.e_ind_probs] #actually logits not probs, feed to softmax for probs

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

    def make_eval_batch(self, dataset, sample_size):
        batch_size = sample_size
        train_q, train_p, train_span, val_q, val_p, val_span = dataset
        start_index = 0 #TODO: make this random sampling later

        #make padded q batch
        qs = val_q[start_index:start_index+batch_size]
        q_seq_lens = np.array([len(q) for q in qs])
        q_seq_mlen = np.max(q_seq_lens)
        q_batch = np.array([q + [PAD_ID]*(q_seq_mlen - len(q)) for q in qs])

        #make padded p batch
        ps = val_p[start_index:start_index+batch_size]
        p_seq_lens = np.array([len(p) for p in ps])
        p_seq_mlen = np.max(p_seq_lens)
        p_batch = np.array([p + [PAD_ID]*(p_seq_mlen - len(p)) for p in ps])

        #make one-hot start and end labels
        spans = val_span[start_index:start_index+batch_size]
        s_inds, e_inds = zip(*spans)
        s_inds = np.array(s_inds)
        e_inds = np.array(e_inds)
        starts = np.eye(p_seq_mlen)[s_inds]
        ends = np.eye(p_seq_mlen)[e_inds]

        return q_batch, q_seq_lens, p_batch, p_seq_lens, starts, ends


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

        q_batch, q_lens, p_batch, p_lens, s_label_batch, e_label_batch = self.make_eval_batch(dataset, sample)

        test_x = (q_batch, q_lens, p_batch, p_lens)
        pred_s, pred_e = self.answer(session, test_x)

        pred_word_inds = [p_batch[i][pred_s[i]:pred_e[i]+1] for i in range(sample)]
        label_word_inds = [p_batch[i][np.argmax(s_label_batch[i]):np.argmax(e_label_batch[i])+1] for i in range(sample)]
        #print(label_word_inds)
        #print(pred_word_inds)

        pred_words = [" ".join(map(str, pred_inds)) for pred_inds in pred_word_inds]
        label_words = [" ".join(map(str, label_inds)) for label_inds in label_word_inds]
        print(pred_words)
        print(label_words)


        f1s = [f1_score(pred_words[i], label_words[i]) for i in range(sample)]
        ems = [exact_match_score(pred_words[i], label_words[i]) for i in range(sample)]

        f1 = sum(f1s)/float(len(f1s))
        em = sum(ems)/float(len(ems))


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

        #make one-hot start and end labels
        spans = train_span[start_index:start_index+batch_size]
        s_inds, e_inds = zip(*spans)
        s_inds = np.array(s_inds)
        e_inds = np.array(e_inds)
        starts = np.eye(p_seq_mlen)[s_inds]
        ends = np.eye(p_seq_mlen)[e_inds]

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


        #run main training loop: (only 10 epochs for now)
        train_q, train_p, train_span, val_q, val_p, val_span = dataset
        print(len(train_q))
        max_iters = np.ceil(len(train_q)/float(self.FLAGS.batch_size))
        print("Max iterations: " + str(max_iters))
        for epoch in range(10):
            #temp hack to only train on some small subset:
            #max_iters = some small constant

            for iteration in range(int(max_iters)):
                print("Current iteration: " + str(iteration))
                q_batch, q_lens, p_batch, p_lens, s_label_batch, e_label_batch = self.make_batch(dataset, iteration)
                lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
                #TODO: set annealed lr?
                #retrieve useful info from training - see optimize() function to set what we're tracking
                _, loss, pred_s, pred_e, label_e, p_mask, grad_norm, attn = self.optimize(session, (q_batch, q_lens, p_batch, p_lens), (s_label_batch, e_label_batch))
                print("Current Loss: " + str(loss))
                #print(pred_s)
                #print(pred_e)
                #print(attn)
                print(grad_norm)
