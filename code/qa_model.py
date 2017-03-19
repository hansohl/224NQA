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
        self.s_labels_placeholder = tf.placeholder(tf.int32, shape=[None], name="s_place") #batch
        self.e_labels_placeholder = tf.placeholder(tf.int32, shape=[None], name="e_place")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=[], name="dropout_place")

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
        #gradient clipping to max
        grads, _ = tf.clip_by_global_norm(grads, self.FLAGS.max_gradient_norm)
        grads_and_vars = [(grads[i], grads_and_vars[i][1]) for i in range(len(grads_and_vars))]
        self.grad_norm = tf.global_norm(grads)
        tf.summary.scalar('grad_norm', self.grad_norm)
        self.training_op = self.optimizer.apply_gradients(grads_and_vars)
        #default (boring!) trainingop
        #self.training_op = self.optimizer.minimize(self.loss)

        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()




    def setup_system(self, encoder, decoder):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        inputs = (self.distr_q, self.distr_p)

        encoding = encoder.encode(inputs, (self.q_mask_placeholder, self.p_mask_placeholder), None)
        self.s_ind_probs, self.e_ind_probs = decoder.decode(encoding, self.p_mask_placeholder, self.dropout_placeholder)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            s_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.s_ind_probs, self.s_labels_placeholder)
            e_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.e_ind_probs, self.e_labels_placeholder)
            s_loss = tf.reduce_mean(s_losses)
            e_loss = tf.reduce_mean(e_losses)
            self.loss = s_loss + e_loss
            tf.summary.scalar('s_loss', s_loss)
            tf.summary.scalar('e_loss', e_loss)
            tf.summary.scalar('loss', self.loss)

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
        input_feed[self.dropout_placeholder] = 1 - self.FLAGS.dropout

        #set the quantities we track/return during training
        output_feed = [self.training_op, self.loss, self.e_ind_probs, self.e_labels_placeholder, self.grad_norm, self.summary]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        q, q_lens, p, p_lens = valid_x
        s_labels, e_labels = valid_y
        input_feed = {}
        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed[self.q_placeholder] = q
        input_feed[self.q_mask_placeholder] = q_lens
        input_feed[self.p_placeholder] = p
        input_feed[self.p_mask_placeholder] = p_lens
        input_feed[self.s_labels_placeholder] = s_labels
        input_feed[self.e_labels_placeholder] = e_labels
        input_feed[self.dropout_placeholder] = 1.0

        #set the quantities we track/return during training
        output_feed = [self.loss]

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
        input_feed[self.dropout_placeholder] = 1.0
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

    def validate(self, session, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0.

        val_q, val_p, val_span = valid_dataset
        max_iters = np.ceil(len(val_q)/float(self.FLAGS.batch_size))
        print("VALIDATING")
        print("Validation iterations: " + str(max_iters))
        for iteration in range(int(max_iters)):
            print("val iteration: " + str(iteration))
            q_batch, q_lens, p_batch, p_lens, s_label_batch, e_label_batch = self.make_validation_batch(valid_dataset, iteration)
            #retrieve useful info from training - see test() function to set what we're tracking
            [loss] = self.test(session, (q_batch, q_lens, p_batch, p_lens), (s_label_batch, e_label_batch))
            valid_cost += loss
        valid_cost = valid_cost / max_iters
        return valid_cost

    def make_eval_batch(self, dataset, sample_size):
        batch_size = sample_size
        train_q, train_p, train_span, val_q, val_p, val_span = dataset
        # start_index = 0

        length = len(val_q) #whatever call this needs to be to know how many samples there are to choose from
        rand_indices = np.random.randint(0, length, size=batch_size)
        # print(rand_indices)
        #make padded q batch
        qs = np.take(val_q, rand_indices, axis=0)#val_q[start_index:start_index+batch_size]
        q_seq_lens = np.array([len(q) for q in qs])
        q_seq_mlen = np.max(q_seq_lens)
        q_batch = np.array([q + [PAD_ID]*(q_seq_mlen - len(q)) for q in qs])

        #make padded p batch
        ps = np.take(val_p, rand_indices, axis=0)#val_p[start_index:start_index+batch_size]
        p_seq_lens = np.array([len(p) for p in ps])
        p_seq_mlen = np.max(p_seq_lens)
        p_batch = np.array([p + [PAD_ID]*(p_seq_mlen - len(p)) for p in ps])

        #make start and end labels
        spans = np.take(val_span, rand_indices, axis=0)#val_span[start_index:start_index+batch_size]
        s_inds, e_inds = zip(*spans)
        starts = np.array(s_inds, dtype=np.int32)
        ends = np.array(e_inds, dtype=np.int32)

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
        label_word_inds = [p_batch[i][s_label_batch[i]:e_label_batch[i]+1] for i in range(sample)]

        pred_words = [" ".join(map(str, pred_inds)) for pred_inds in pred_word_inds]
        label_words = [" ".join(map(str, label_inds)) for label_inds in label_word_inds]


        f1s = [f1_score(pred_words[i], label_words[i]) for i in range(sample)]
        ems = [exact_match_score(pred_words[i], label_words[i]) for i in range(sample)]

        f1 = sum(f1s)/float(len(f1s))
        em = sum(ems)/float(len(ems))


        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
        return f1, em


    def make_validation_batch(self, valid_dataset, iteration):
        batch_size = self.FLAGS.batch_size
        val_q, val_p, val_span = valid_dataset

        start_index = iteration*batch_size
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

        #make start and end labels
        spans = val_span[start_index:start_index+batch_size]
        s_inds, e_inds = zip(*spans)
        starts = np.array(s_inds, dtype=np.int32)
        ends = np.array(e_inds, dtype=np.int32)

        return q_batch, q_seq_lens, p_batch, p_seq_lens, starts, ends

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

        #make start and end labels
        spans = train_span[start_index:start_index+batch_size]
        s_inds, e_inds = zip(*spans)
        starts = np.array(s_inds, dtype=np.int32)
        ends = np.array(e_inds, dtype=np.int32)

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

        summary_writer = tf.summary.FileWriter(self.FLAGS.log_dir, session.graph)

        #run main training loop: (only 10 epochs for now)
        train_q, train_p, train_span, val_q, val_p, val_span = dataset
        valid_dataset = (val_q, val_p, val_span)
        max_iters = np.ceil(len(train_q)/float(self.FLAGS.batch_size))
        print("Max iterations: " + str(max_iters))
        for epoch in range(10):
            #temp hack to only train on some small subset:
            #max_iters = some small constant

            for iteration in range(int(max_iters)):
                print("Current iteration: " + str(iteration))
                q_batch, q_lens, p_batch, p_lens, s_label_batch, e_label_batch = self.make_batch(dataset, iteration)
                #lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
                #TODO: set annealed lr?
                #retrieve useful info from training - see optimize() function to set what we're tracking
                _, loss, pred_e, label_e, grad_norm, summ_str = self.optimize(session, (q_batch, q_lens, p_batch, p_lens), (s_label_batch, e_label_batch))
                print("Current Loss: " + str(loss))
                print(grad_norm)

                summary_writer.add_summary(summ_str, iteration)
                #eval on first 100 in val set every 100 iterations
                if iteration%100==99:
                    self.evaluate_answer(session, dataset, log=True)
                if iteration%400==399:
                    valid_loss = self.validate(session, valid_dataset)
                    print("Validation Loss: " + str(valid_loss))

            #done with epoch
            save_path = self.saver.save(session, self.FLAGS.train_dir + "/model_ep" + str(epoch) + ".ckpt")
            logging.info("Model saved in file {}".format(save_path))
