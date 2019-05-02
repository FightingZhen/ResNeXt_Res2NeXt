import sys

sys.path.append('../Data_Initialization/')
import tensorflow as tf
import tensorflow.contrib.layers as layers
from cifar100 import *
import evaluation_function as eval
import time


class ResNeXt(object):
    def __init__(self, model_name, sess, train_data, tst_data, epoch, num_class, ksize, weight_decay, momentum,
                 cardinality, width, block_num1, block_num2, block_num3, learning_rate, batch_size, img_height,
                 img_width):

        self.sess = sess
        self.training_data = train_data
        self.test_data = tst_data
        self.eps = epoch
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.k = ksize
        self.wd = weight_decay
        self.momentum = momentum
        self.c = cardinality
        self.w = width
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.block_num3 = block_num3
        self.lr = learning_rate
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class

        self.oc1 = self.c * self.w
        self.oc2 = self.oc1 * 2
        self.oc3 = self.oc2 * 2

        self.build_model()
        self.saveConfiguration()

    def saveConfiguration(self):
        save2file('model : %s' % self.model, self.ckptDir, self.model)
        save2file('num class : %d' % self.num_class, self.ckptDir, self.model)
        save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        save2file('ksize : %d' % self.k, self.ckptDir, self.model)
        save2file('weight decay : %g' % self.wd, self.ckptDir, self.model)
        save2file('momentum : %g' % self.momentum, self.ckptDir, self.model)
        save2file('cardinality : %d' % self.c, self.ckptDir, self.model)
        save2file('width : %d' % self.w, self.ckptDir, self.model)
        save2file('out channel 1 : %d' % self.oc1, self.ckptDir, self.model)
        save2file('out channel 2 : %d' % self.oc2, self.ckptDir, self.model)
        save2file('out channel 3 : %d' % self.oc3, self.ckptDir, self.model)
        save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        save2file('image width : %d' % self.img_w, self.ckptDir, self.model)

    def conv(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_result = tf.layers.conv2d(inputs=inputMap, filters=out_channel, kernel_size=(ksize, ksize),
                                           strides=(stride, stride), padding=padding, use_bias=False,
                                           kernel_initializer=layers.variance_scaling_initializer(), name='conv')

            return conv_result

    def bn(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training)

    def relu(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def avgPool(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    def globalAvgPool(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            size = inputMap.get_shape()[1]
            return self.avgPool(inputMap, size, size, padding='VALID', scope_name=scope_name)

    def flatten(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.layers.flatten(inputMap)

    def concatenation(self, inputMapList, axis, scope_name):
        with tf.variable_scope(scope_name):
            return tf.concat(inputMapList, axis)

    def linear(self, inputMap, out_channel, scope_name, use_bias):
        with tf.variable_scope(scope_name):
            linear_result = tf.layers.dense(inputs=inputMap, units=out_channel,
                                            kernel_initializer=layers.variance_scaling_initializer(), use_bias=use_bias,
                                            name='linear')

            return linear_result

    def first_layer(self, inputMap, out_channel, is_training, scope_name):
        with tf.variable_scope(scope_name):
            _conv = self.conv(inputMap, out_channel=out_channel, ksize=self.k, stride=1, scope_name='_conv')
            _bn = self.bn(_conv, scope_name='_bn', is_training=is_training)
            _relu = self.relu(_bn, scope_name='_relu')
            return _relu

    def transition_layer(self, inputMap, out_channel, scope_name, is_training, use_relu):
        with tf.variable_scope(scope_name):
            _conv = self.conv(inputMap, out_channel=out_channel, ksize=1, stride=1, scope_name='_conv')
            _bn = self.bn(_conv, is_training=is_training, scope_name='_bn')
            if use_relu:
                _result = self.relu(_bn, scope_name='_relu')
            else:
                _result = _bn

            return _result

    def groupConv_layer(self, inputMap, ksize, stride, group, scope_name, is_training):
        with tf.variable_scope(scope_name):
            in_dim = int(np.shape(inputMap)[-1])
            group_dim = in_dim // group

            featureMaps_list = list()
            for i in range(group):
                _featureMap = inputMap[:, :, :, i * group_dim:(i + 1) * group_dim]
                _conv = self.conv(_featureMap, out_channel=group_dim, ksize=ksize, stride=stride,
                                  scope_name='_conv_group' + str(i + 1))
                featureMaps_list.append(_conv)

            concatenated = self.concatenation(featureMaps_list, axis=3, scope_name='_concatenated')
            _bn = self.bn(concatenated, is_training=is_training, scope_name='_bn')
            _relu = self.relu(_bn, scope_name='_relu')

            return _relu

    def residual_block(self, inputMap, ksize, out_channel, cardinality, scope_name, is_training, first_block):
        with tf.variable_scope(scope_name):
            input_dim = int(np.shape(inputMap)[-1])

            if input_dim == out_channel * 4:
                stride = 1
                flag = False
            elif first_block:
                stride = 1
                pad_channel = int(out_channel * 4 - input_dim) // 2
                flag = True
            else:
                stride = 2
                pad_channel = int(out_channel * 4 - input_dim) // 2
                flag = True

            _fuse_layer = self.transition_layer(inputMap, out_channel=out_channel, scope_name='_fuse_layer',
                                                is_training=is_training, use_relu=True)
            _groupConv_layer = self.groupConv_layer(_fuse_layer, ksize=ksize, stride=stride, group=cardinality,
                                                    scope_name='_groupConv_layer', is_training=is_training)
            _expand_layer = self.transition_layer(_groupConv_layer, out_channel=out_channel * 4,
                                                  scope_name='_expand_layer', is_training=is_training, use_relu=False)

            if flag:
                padding = [[0, 0], [0, 0], [0, 0], [pad_channel, pad_channel]]
                identity_map = inputMap
                if not first_block:
                    identity_map = self.avgPool(identity_map, ksize=2, stride=2, scope_name='identity_avgpool')
                identity_map = tf.pad(identity_map, paddings=padding, name='_identity_padding')
            else:
                identity_map = inputMap

            _added = tf.add(_expand_layer, identity_map, name='_added')
            _final_relu = self.relu(_added, scope_name='_final_relu')

            return _final_relu

    def residual_stage(self, inputMap, ksize, out_channel, cardinality, block_num, stage_name, is_training,
                       first_stage):
        with tf.variable_scope(stage_name):
            _block = self.residual_block(inputMap, ksize=ksize, out_channel=out_channel, cardinality=cardinality,
                                         scope_name='_block1', is_training=is_training, first_block=first_stage)
            for i in range(2, block_num + 1):
                _block = self.residual_block(_block, ksize=ksize, out_channel=out_channel, cardinality=cardinality,
                                             scope_name='_block' + str(i), is_training=is_training, first_block=False)

            return _block

    def resnext_model(self, inputMap, model_name, ksize, cardinality, block_num1, block_num2, block_num3, out_channel1,
                      out_channel2, out_channel3, is_training, reuse):
        with tf.variable_scope(model_name, reuse=reuse):
            _first_layer = self.first_layer(inputMap, out_channel=64, is_training=is_training,
                                            scope_name='_first_layer')

            stage1 = self.residual_stage(inputMap=_first_layer,
                                         ksize=ksize,
                                         out_channel=out_channel1,
                                         cardinality=cardinality,
                                         block_num=block_num1,
                                         stage_name='_stage1',
                                         is_training=is_training,
                                         first_stage=True)

            stage2 = self.residual_stage(inputMap=stage1,
                                         ksize=ksize,
                                         out_channel=out_channel2,
                                         cardinality=cardinality,
                                         block_num=block_num2,
                                         stage_name='_stage2',
                                         is_training=is_training,
                                         first_stage=False)

            stage3 = self.residual_stage(inputMap=stage2,
                                         ksize=ksize,
                                         out_channel=out_channel3,
                                         cardinality=cardinality,
                                         block_num=block_num3,
                                         stage_name='_stage3',
                                         is_training=is_training,
                                         first_stage=False)

            _globalPool = self.globalAvgPool(stage3, scope_name='_globalPool')
            _flatten = self.flatten(_globalPool, scope_name='_flatten')

            y_pred = self.linear(_flatten, self.num_class, scope_name='linear', use_bias=True)
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.img_h, self.img_w, 3])
        self.y = tf.placeholder(tf.int32, [None, self.num_class])
        self.is_training = tf.placeholder(tf.bool)
        tf.summary.image('Image/Origin', self.x, max_outputs=5)

        self.y_pred, self.y_pred_softmax = self.resnext_model(inputMap=self.x,
                                                              model_name=self.model,
                                                              ksize=self.k,
                                                              cardinality=self.c,
                                                              block_num1=self.block_num1,
                                                              block_num2=self.block_num2,
                                                              block_num3=self.block_num3,
                                                              out_channel1=self.oc1,
                                                              out_channel2=self.oc2,
                                                              out_channel3=self.oc3,
                                                              is_training=self.is_training,
                                                              reuse=False)

        with tf.variable_scope('loss'):
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y))
            self.l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            self.loss = self.cost + self.wd * self.l2_loss
            tf.summary.scalar('Loss/cost', self.cost)
            tf.summary.scalar('Loss/L2_loss', self.l2_loss)
            tf.summary.scalar('Loss/Total loss', self.loss)

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.MomentumOptimizer(self.lr, momentum=self.momentum).minimize(self.loss)

        with tf.variable_scope('tfSummary'):
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.ckptDir, self.sess.graph)

        with tf.variable_scope('saver'):
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.eps)

        with tf.variable_scope('accuracy'):
            self.distribution = [tf.argmax(self.y, 1), tf.argmax(self.y_pred_softmax, 1)]
            self.correct_prediction = tf.equal(self.distribution[0], self.distribution[1])
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.itr_epoch = len(self.training_data[0]) // self.bs

        training_acc = 0.0
        training_loss = 0.0

        for e in range(1, self.eps + 1):
            if e == 150 or e == 225:
                self.lr *= 0.1
            for itr in range(self.itr_epoch):
                _index = np.random.randint(low=0, high=len(self.training_data[0]), size=self.bs)
                _tr_img_batch = self.training_data[0][_index]
                _tr_lab_batch = self.training_data[1][_index]

                _tr_img_batch = data_augmentation(_tr_img_batch)

                _train_accuracy, _train_loss, _ = self.sess.run([self.accuracy, self.cost, self.train_op],
                                                                feed_dict={self.x: _tr_img_batch,
                                                                           self.y: _tr_lab_batch,
                                                                           self.is_training: True})
                training_acc += _train_accuracy
                training_loss += _train_loss

            summary = self.sess.run(self.merged, feed_dict={self.x: _tr_img_batch,
                                                            self.y: _tr_lab_batch,
                                                            self.is_training: False})

            training_acc = float(training_acc / self.itr_epoch)
            training_loss = float(training_loss / self.itr_epoch)

            log = "Epoch: [%d], Training Accuracy: [%g], Training Loss: [%g], Learning Rate: [%f], Time: [%s]" % \
                  (e, training_acc, training_loss, self.lr, time.ctime(time.time()))

            save2file(log, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))

            eval.test_procedure(test_data=self.test_data, distribution_op=self.distribution, inputX=self.x,
                                inputY=self.y, mode='test', num_class=self.num_class, batch_size=self.bs,
                                session=self.sess, is_training=self.is_training, ckptDir=self.ckptDir,
                                model=self.model)

            training_acc = 0.0
            training_loss = 0.0
