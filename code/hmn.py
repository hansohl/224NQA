POOL_SIZE = 16
HIDDEN_LAYER_SIZE = 200
batch_size = 10
question_length = 600

def HMN(U, h_i, s_prev, e_prev, scope_name):
    with tf.variable_scope(scope_name) as scope:
        # u_s = U[:,s_prev]
        # u_e = U[:,e_prev]

        W1 = tf.get_variable("HMN_W1", [3 * HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, POOL_SIZE], initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable("HMN_W2", [HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, POOL_SIZE], initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable("HMN_W3", [2*HIDDEN_LAYER_SIZE, 1, POOL_SIZE], initializer=tf.contrib.layers.xavier_initializer())
        Wd = tf.get_variable("HMN_WD", [5 * HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable("HMN_B1", tf.zeros([HIDDEN_LAYER_SIZE, POOL_SIZE]))
        b2 = tf.get_variable("HMN_B2", tf.zeros([HIDDEN_LAYER_SIZE, POOL_SIZE]))
        b3 = tf.get_variable("HMN_B3", tf.zeros([POOL_SIZE,]))

        h_i_tiled = tf.tile(h_i, (1, batch_size)) #h is indeed a batch
        hsu = tf.concat(1, [h_i_tiled, s_prev, e_prev])
        #hsu is batch_size x (hidden_size * 5)
        r = tf.nn.tanh(tf.matmul(hsu, Wd)) # r is batch_size x hidden_size
        r_tiled = tf.tile(r, (1, m)) #U is batch_size x question_length x (hidden_size * 2)
        Ur = tf.concat(U, r_tiled) #Ur is batch_size x question_length x (hidden_size * 3)
        m1 = tf.reduce_max(tf.matmul(Ur, W1) + b1, axis=3) #m1 is batch_size x question_length x hidden_size (the pool size was maxed over)
        m2 = tf.reduce_max(tf.matmul(m1, W2) + b2, axis=3) #m2 is batch_size x question_length x hidden_size (pool maxed over again)

        m1m2 = tf.concat(2, [m1, m2]) #m1m2 is batch_size x question_length x (2 * hidden_size)
        hmn = tf.reduce_max(tf.matmul(m1m2, W3) + b3, axis=0)
        #before_max, we have batch_size x question_length x 1 x POOL_SIZE
        # after, we have batch_size x question_length (these are the alpha t's)

        new_u = tf.argmax(hmn, axis=0)

        return new_u, U[:,new_u]
