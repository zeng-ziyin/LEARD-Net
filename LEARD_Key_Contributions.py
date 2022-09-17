    def ResiDense(self, feature, xyzrgb, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_pc_1 = self.LFEA(feature, xyzrgb, neigh_idx, d_out // 4, name + 'LFEA_1', is_training)  # B, N, 1, d_out/4
        f_pc_2 = self.LFEA(f_pc_1, xyzrgb, neigh_idx, d_out // 4, name + 'LFEA_2', is_training)  # B, N, 1, d_out/4
        f_pc_3 = self.LFEA(tf.concat([f_pc_1, f_pc_2], axis=-1), xyzrgb, neigh_idx, d_out // 2, name + 'LFEA_3', is_training)  # B, N, 1, d_out/2
        f_layer_out = tf.concat([f_pc_1, f_pc_2, f_pc_3], axis=-1)  # B, N, 1, d_out   Dense Connection
        shortcut = helper_tf_util.conv2d(feature, d_out, [1, 1], name + 'residual', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        f_pc = helper_tf_util.conv2d(f_layer_out + shortcut, d_out, [1, 1], name + 'final_mlp', [1, 1], 'VALID', True, is_training, activation_fn=None)

        return tf.nn.leaky_relu(f_pc)

    def LFEA(self, feature, xyzrgb, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        feature = tf.squeeze(feature, axis=2)

        aug_xyzrgb = self.loc_sc(xyzrgb, neigh_idx)
        aug_feat = self.loc_feat(feature, neigh_idx)

        BF = self.loc_sc(xyzrgb, neigh_idx)  # Bilateral Feature Encoding
        m_BF = helper_tf_util.conv2d(aug_xyzrgb, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        MF = self.loc_feat(feature, neigh_idx, is_k=False)  # Multidimensional Feature Encoding
        m_MF = helper_tf_util.conv2d(aug_feat, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        new_BF = helper_tf_util.conv2d(tf.concat([BF, m_MF], axis=-1), d_out // 2, [1, 1], name + 'mlp3', [1, 1], 'VALID', True, is_training)
        new_MF = helper_tf_util.conv2d(tf.concat([MF, m_BF], axis=-1), d_out // 2, [1, 1], name + 'mlp4', [1, 1], 'VALID', True, is_training)
        CF = tf.concat([new_BF, new_MF], axis=-1)  # Cross Feature Encoding

        scope = helper_tf_util.conv2d(CF, d_out, [1, 1], name + 'mlp5', [1, 1], 'VALID', bn=False, activation_fn=None)  # B, N, k, d_out/2
        scope = tf.nn.softmax(scope, axis=2)  # B, N, k, d_out/2
        attention_pooling = tf.reduce_sum(CF * scope, axis=2, keepdims=True)  # B, N, 1, d_out/2
        max_pooling = tf.reduce_max(CF, axis=2, keepdims=True)  # B, N, 1, d_out/2
        UP = tf.concat([max_pooling, attention_pooling], axis=-1)  # B, N, 1, d_out   United Pooling
        UP = helper_tf_util.conv2d(UP, d_out, [1, 1], name + 'mlp6', [1, 1], 'VALID', bn=False, activation_fn=None)  # B, N, k, d_out/2

        return UP

    def loc_feat(self, x, neigh_idx, is_k=True):
        neighbor_x = self.gather_neighbour(x, neigh_idx)
        x_tile = tf.tile(tf.expand_dims(x, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        if is_k:
            x_info = x_tile - neighbor_x  # B, N, k, d_in
            x_info = -tf.abs(x_info)
        else:
            x_info = tf.concat([x_tile, neighbor_x, x_tile - neighbor_x], axis=-1)  # B, N, k, 3*d_in

        return x_info

    def loc_sc(self, xyzrgb, neigh_idx):
        tile_xyzrgb = tf.tile(tf.expand_dims(xyzrgb, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])  # B, N, k, 6
        neigh_xyzrgb = self.gather_neighbour(xyzrgb, neigh_idx)  # B, N, k, 6

        Grayscale_in = tf.reduce_sum(tile_xyzrgb[..., 3:6], -1, keep_dims=True)
        Grayscale_knn = tf.reduce_sum(neigh_xyzrgb[..., 3:6], -1, keep_dims=True)
        delta_Color = Grayscale_in - Grayscale_knn

        relative_xyz = tile_xyzrgb[..., :3] - neigh_xyzrgb[..., :3]
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))

        xyzrgb_info = tf.concat([tile_xyzrgb, neigh_xyzrgb, tile_xyzrgb - neigh_xyzrgb, relative_dis, delta_Color], axis=-1)

        return xyzrgb_info  # B, N, k, 6+6+6+2
