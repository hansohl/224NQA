import tensorflow as tf

POOL_SIZE = 16
HIDDEN_LAYER_SIZE = 200
# batch_size = 10
# question_length = 600

def HMN(U, h_i, s_prev, e_prev, iteration, scope_name):
    with tf.variable_scope(scope_name) as scope:
        if iteration > 0:
            scope.reuse_variables()
        # u_s = U[:,s_prev]
        # u_e = U[:,e_prev]
        batch_size = tf.shape(U)[0]
        document_length = tf.shape(U)[1]

        W1 = tf.get_variable("HMN_W1", [3 * HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE * POOL_SIZE], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        W2 = tf.get_variable("HMN_W2", [HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE * POOL_SIZE], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        W3 = tf.get_variable("HMN_W3", [2*HIDDEN_LAYER_SIZE, 1 * POOL_SIZE], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        Wd = tf.get_variable("HMN_WD", [5 * HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

        b1 = tf.get_variable("HMN_B1", [HIDDEN_LAYER_SIZE * POOL_SIZE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        b2 = tf.get_variable("HMN_B2", [HIDDEN_LAYER_SIZE * POOL_SIZE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        b3 = tf.get_variable("HMN_B3", [POOL_SIZE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

        # h_i_tiled = tf.tile(h_i, (batch_size, 1)) #h is indeed a batch
        hsu = tf.concat(1, [h_i, s_prev, e_prev])
        #hsu is batch_size x (hidden_size * 5)
        r = tf.nn.tanh(tf.matmul(hsu, Wd)) # r is batch_size x hidden_size
        r = tf.expand_dims(r, 1) # batch_size x 1 x hidden_size
        r_tiled = tf.tile(r, (1, document_length, 1)) #U is batch_size x document_length x (hidden_size * 2)
        Ur = tf.concat(2, [U, r_tiled]) #Ur is batch_size x question_length x (hidden_size * 3)
        Ur = tf.reshape(Ur, [batch_size * document_length, HIDDEN_LAYER_SIZE * 3])
        m1_inner = tf.matmul(Ur, W1) + b1
        m1_inner = tf.reshape(m1_inner, [batch_size, document_length, HIDDEN_LAYER_SIZE, POOL_SIZE])
        m1 = tf.reduce_max(m1_inner, axis=3) #m1 is batch_size x document_length x hidden_size (the pool size was maxed over)
        m1_r = tf.reshape(m1, [batch_size * document_length, HIDDEN_LAYER_SIZE])
        m2_inner = tf.matmul(m1_r, W2) + b2
        m2_inner = tf.reshape(m2_inner, [batch_size, document_length, HIDDEN_LAYER_SIZE, POOL_SIZE])
        m2 = tf.reduce_max(m2_inner, axis=3) #m2 is batch_size x document_length x hidden_size (pool maxed over again)

        m1m2 = tf.concat(2, [m1, m2]) #m1m2 is batch_size x document_length x (2 * hidden_size)
        m1m2 = tf.reshape(m1m2, [batch_size * document_length, 2 * HIDDEN_LAYER_SIZE])
        hmn_inner = tf.matmul(m1m2, W3) + b3
        hmn_inner = tf.reshape(hmn_inner, [batch_size, document_length, 1, POOL_SIZE])
        hmn = tf.reduce_max(hmn_inner, axis=3)
        #before_max, we have batch_size x document_length x 1 x POOL_SIZE
        # after, we have batch_size x document_length x 1 (these are the alpha t's)

        # new_u = tf.argmax(hmn, axis=0)
        batch_size = tf.shape(U, out_type=tf.int64)[0]

        alpha_beta = tf.squeeze(hmn)
        new_u = tf.argmax(alpha_beta, axis=1) # i think since we want the max over the question embedding
        new_u = tf.expand_dims(new_u, 1)
        # new u is just size [batch_size]
        batch_range = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), 1)

        ind = tf.concat(1, [batch_range, new_u])
        new_u_vec = tf.gather_nd(U, ind)

        # scope.reuse_variables()

        return new_u, new_u_vec, alpha_beta
