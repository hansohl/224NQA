POOL_SIZE = 16
HIDDEN_LAYER_SIZE = 200

def HMN(U, h_i, s_prev, e_prev, scope_name):
    with tf.variable_scope(scope_name) as scope:
        # u_s = U[:,s_prev]
        # u_e = U[:,e_prev]

        W1 = tf.get_variable("HMN_W1", [POOL_SIZE, HIDDEN_LAYER_SIZE, 3*HIDDEN_LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable("HMN_W2", [POOL_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable("HMN_W3", [POOL_SIZE, 1, 2*HIDDEN_LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer())
        Wd = tf.get_variable("HMN_WD", [HIDDEN_LAYER_SIZE, 5*HIDDEN_LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable("HMN_B1", tf.zeros([POOL_SIZE, HIDDEN_LAYER_SIZE]))
        b2 = tf.get_variable("HMN_B2", tf.zeros([POOL_SIZE, HIDDEN_LAYER_SIZE]))
        b3 = tf.get_variable("HMN_B3", tf.zeros(POOL_SIZE))

        h_i_tiled = tf.tile(h_i, (1, batch_size)) #WHAT IS THE BATCH SIZE? Is h_i already a batch?
        hsu = tf.concat(1, [h_i_tiled, s_prev, e_prev])

        r = tf.nn.tanh(tf.matmul(hsu, Wd))
        r_tiled = tf.tile(r, (1, m))
        Ur = tf.concat(U, r_tiled)
        m1 = tf.reduce_max(tf.matmul(Ur, W1) + b1, axis=0)
        m2 = tf.reduce_max(tf.matmul(m1, W2) + b2, axis=0)

        m1m2 = tf.concat(1, [m1, m2])
        hmn = tf.reduce_max(tf.matmul(m1m2, W3) + b3, axis=0)

        new_u = tf.argmax(hmn, axis=0)

        return new_u, U[:,new_u]
