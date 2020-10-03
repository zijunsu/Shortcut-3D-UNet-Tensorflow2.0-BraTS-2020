import tensorflow as tf
#fix bug
tf.random.Generator = None
import tensorflow_addons as tfa
import utils
import gradient_checkpointing
import checkpointing

l2 = 1e-4
drop_prob = 0.0
groups = 4
sample = False
shortcut = True
# l2 = 0
# drop_prob = 0
class conv_layers(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 stride=(1, 1, 1),
                 kernel_size=(3, 3, 3),
                 pool_size=(2, 2, 2),
                 padding='SAME',
                 kernel_regularizers=tf.keras.regularizers.l1_l2(0, l2),
                 activation_fn=tf.nn.relu,
                 last=False,
                 drop_prob=drop_prob,
                 **kwargs):

        super(conv_layers, self).__init__(**kwargs)

        # inter_filters = int(filters/2)
        self.conv_1 = tf.keras.layers.Conv3D(filters, kernel_size, stride, padding,
                                             kernel_regularizer=kernel_regularizers)
        self.conv_2 = tf.keras.layers.Conv3D(filters, kernel_size, stride, padding,
                                             kernel_regularizer=kernel_regularizers)
        if groups != 0:
            self.gn_1 = tfa.layers.GroupNormalization(groups=groups)
            self.gn_2 = tfa.layers.GroupNormalization(groups=groups)
        else:
            self.gn_1 = tfa.layers.InstanceNormalization()
            self.gn_2 = tfa.layers.InstanceNormalization()

        self.activation_fn_1 = activation_fn
        self.activation_fn_2 = activation_fn
        self.last = last

        if not self.last:
            self.pool = tf.keras.layers.MaxPool3D(pool_size, 2, 'SAME')
        if shortcut:
            self.shortcut_ocnv = tf.keras.layers.Conv3D(filters, 1, stride, padding,
                                                        kernel_regularizer=kernel_regularizers)
        # else:
        #     self.dp_1 = tf.keras.layers.SpatialDropout3D(drop_prob)
        #     self.dp_2 = tf.keras.layers.SpatialDropout3D(drop_prob)

    # @checkpointing.checkpointable
    def call(self, inputs, **kwargs):
        p_x = tf.ones(1)
        x = self.conv_1(inputs)
        x = self.gn_1(x)
        x = self.activation_fn_1(x)
        # if self.last:
        #     x = self.dp_1(x)

        x = self.conv_2(x)
        x = self.gn_2(x)
        if shortcut:
            shortcut_x = self.shortcut_ocnv(inputs)
            x += shortcut_x
        x = self.activation_fn_2(x)
        if not self.last:
            p_x = self.pool(x)
        # else:
        #     x = self.dp_2(x)

        return x, p_x

    def model(self, shape):
        x = tf.keras.Input(shape)
        return tf.keras.Model(inputs=[x], outputs=[self.call(x)])

class attention(tf.keras.layers.Layer):
    def __init__(self,F_init,
                 kernel_regularizers=tf.keras.regularizers.l1_l2(0, l2)):
        super(attention, self).__init__()

        self.conv_g = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(F_init, (1, 1), padding='same', kernel_regularizer=kernel_regularizers),
             tf.keras.layers.BatchNormalization()])
        self.conv_x = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(F_init, (1, 1), padding='same', kernel_regularizer=kernel_regularizers),
             tf.keras.layers.BatchNormalization()])
        self.att_conv = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(1, (1, 1), padding='same', kernel_regularizer=kernel_regularizers),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Activation(tf.nn.sigmoid)])

    def call(self, g, x):
        f_g = self.conv_g(g)
        f_x = self.conv_x(x)
        fused_features = tf.nn.relu(f_g+f_x)
        prob = self.att_conv(fused_features)

        return tf.concat([g, x*prob], axis=-1)

class deconv_layers(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 layers=0,
                 stride=(1, 1, 1),
                 de_stride=(2, 2, 2),
                 kernel_size=(3, 3, 3),
                 padding='SAME',
                 kernel_regularizers=tf.keras.regularizers.l1_l2(0, l2),
                 activation_fn=tf.nn.relu,
                 sample=sample,
                 is_attention=False,
                 **kwargs):

        super(deconv_layers, self).__init__(**kwargs)

        if sample:
            self.deconv_1 = tf.keras.layers.UpSampling3D(2)
        else:
            self.deconv_1 = tf.keras.layers.Conv3DTranspose(filters, kernel_size, de_stride, padding,
                                                            kernel_regularizer=kernel_regularizers)

        if is_attention:
            self.attention = attention(filters, kernel_regularizers=kernel_regularizers)
        else:
            self.attention = lambda x, y: tf.concat([x, y], axis=-1)

        self.conv_1 = tf.keras.layers.Conv3D(filters, kernel_size, stride, padding,
                                             kernel_regularizer=kernel_regularizers)
        self.conv_2 = tf.keras.layers.Conv3D(filters, kernel_size, stride, padding,
                                             kernel_regularizer=kernel_regularizers)

        if groups != 0:
            self.gn_1 = tfa.layers.GroupNormalization(groups=groups)
            self.gn_2 = tfa.layers.GroupNormalization(groups=groups)
        else:
            self.gn_1 = tfa.layers.InstanceNormalization()
            self.gn_2 = tfa.layers.InstanceNormalization()
        # self.gn_1 = tf.keras.layers.BatchNormalization()
        # self.gn_2 = tf.keras.layers.BatchNormalization()
        self.activation_fn_1 = activation_fn
        self.activation_fn_2 = activation_fn

        self.layers = layers

        if self.layers > 0:
            self.last_conv = tf.keras.layers.Conv3D(self.layers, (1, 1, 1), stride, padding,
                                                    kernel_regularizer=kernel_regularizers)
        if shortcut:
            self.shortcut_ocnv = tf.keras.layers.Conv3D(filters, 1, stride, padding,
                                                        kernel_regularizer=kernel_regularizers)

    def call(self, inputs, concat_inputs=None, **kwargs):
        de_inputs = self.deconv_1(inputs)
        concat_x  = self.attention(de_inputs, concat_inputs)

        x = self.conv_1(concat_x)
        x = self.gn_1(x)
        x = self.activation_fn_1(x)
        x = self.conv_2(x)
        x = self.gn_2(x)
        x = self.activation_fn_2(x)
        if shortcut:
            shortcut_x = self.shortcut_ocnv(concat_x)
            x += shortcut_x

        if self.layers > 0:
            x = self.last_conv(x)
            x = tf.nn.softmax(x, axis=-1)
        return x

    def model(self, shape):
        x = tf.keras.Input(shape)
        return tf.keras.Model(inputs=[x], outputs=[self.call(x)])

class UNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(UNet, self).__init__(**kwargs)

        self.conv_1 = conv_layers(32)
        self.conv_2 = conv_layers(64)
        self.conv_3 = conv_layers(128)
        # last=True, 停止maxpooling
        self.conv_4 = conv_layers(256, last=True)

        # self.deconv_5 = deconv_layers(512)
        self.deconv_3 = deconv_layers(128)
        self.deconv_2 = deconv_layers(64)
        self.deconv_1 = deconv_layers(32, layers=4)

    # @checkpointing.checkpointable
    def call(self, inputs, training=None, mask=None, size=(512, 512)):
        x_1, p_1 = self.conv_1(inputs)
        x_2, p_2 = self.conv_2(p_1)
        x_3, p_3 = self.conv_3(p_2)
        x_4, p_4 = self.conv_4(p_3)
        # x_6, p_6 = self.conv_6(p_5)
        #
        # rx_5 = self.deconv_5(x_6, concat_inputs=x_5)
        rx_3 = self.deconv_3(x_4, concat_inputs=x_3)
        rx_2 = self.deconv_2(rx_3, concat_inputs=x_2)
        rx_1 = self.deconv_1(rx_2, concat_inputs=x_1)

        return rx_1

    def model(self, shape, batch_size=1):
        x = tf.keras.Input(shape, batch_size=1)

        return tf.keras.Model(inputs=[x], outputs=[self.call(x)])


@tf.function
def graph_test(inputs, m):
    # save graph in tensorboard
    m(inputs)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '10'
    tf.summary.trace_on()

    shape = (1, 128, 160, 160, 4)
    inputs = tf.ones(shape)

    m = UNet()
    # simplify summary
    m.model(shape[1:]).summary()
    # tfb graph
    graph_test(inputs, m)
    writer = tf.summary.create_file_writer('summaries/test')
    with writer.as_default():
        tf.summary.trace_export(
            name='tf2_graph',
            step=0
        )
