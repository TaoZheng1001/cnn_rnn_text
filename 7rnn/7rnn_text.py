# _*_ coding:utf-8 _*_

import tensorflow as tf
import os
import sys
import numpy as np
import math
from Vocab import Vocab
from CategoryDict import CategoryDict
from TextDataSet import TextDataSet

tf.logging.set_verbosity(tf.logging.INFO)


def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size=16,
        num_timesteps=50,
        num_lstm_nodes=[32, 32],
        num_lstm_layers=2,
        num_fc_nodes=32,
        batch_size=100,
        clip_lstm_grads=1.0,
        learning_rate=0.001,
        num_word_threshold=10,
    )


hps = get_default_params()

train_file = './data/cnews.train.seg.txt'
val_file = './data/cnews.val.seg.txt'
test_file = './data/cnews.test.seg.txt'
vocab_file = './data/cnews.vocab.txt'
category_file = './data/cnews.category.txt'
output_folder = './run_text_rnn'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

vocab = Vocab(vocab_file, hps.num_word_threshold)
test_str = '的 和 是'
print(vocab.sentence_to_id(test_str))

vocab = Vocab(vocab_file, hps.num_word_threshold)
vocab_size = vocab.size()

category_vocab = CategoryDict(category_file)
num_classes = category_vocab.size()
test_str = '时尚'
tf.logging.info(
    'label: %s, id: %d' % (
        test_str,
        category_vocab.category_to_id(test_str)))

train_dataset = TextDataSet(
    train_file, vocab, category_vocab, hps.num_timesteps)
val_dataset = TextDataSet(
    val_file, vocab, category_vocab, hps.num_timesteps)
test_dataset = TextDataSet(
    test_file, vocab, category_vocab, hps.num_timesteps)


def create_model(hps, vocab_size, num_classes):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    global_step = tf.Variable(
        tf.zeros([], tf.int64), name='global_step', trainable=False)

    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope(
            'embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable(
            'embedding',
            [vocab_size, hps.num_embedding_size],
            tf.float32)
        # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)

    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)

    def _generate_params_for_lstm_cell(x_size, h_size, bias_size):
        """generates parameters for pure lstm implementation."""
        x_w = tf.get_variable('x_weights', x_size)
        h_w = tf.get_variable('h_weights', h_size)
        b = tf.get_variable('biases', bias_size,
                            initializer=tf.constant_initializer(0.0))
        return x_w, h_w, b

    with tf.variable_scope('lstm_nn', initializer=lstm_init):
        """
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(
                hps.num_lstm_nodes[i],
                state_is_tuple = True)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob = keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        initial_state = cell.zero_state(batch_size, tf.float32)
        # rnn_outputs: [batch_size, num_timesteps, lstm_outputs[-1]]
        rnn_outputs, _ = tf.nn.dynamic_rnn(
            cell, embed_inputs, initial_state = initial_state)
        last = rnn_outputs[:, -1, :]
        """
        with tf.variable_scope('inputs'):
            ix, ih, ib = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]]
            )
        with tf.variable_scope('outputs'):
            ox, oh, ob = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]]
            )
        with tf.variable_scope('forget'):
            fx, fh, fb = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]]
            )
        with tf.variable_scope('memory'):
            cx, ch, cb = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]]
            )
        state = tf.Variable(
            tf.zeros([batch_size, hps.num_lstm_nodes[0]]),
            trainable=False
        )
        h = tf.Variable(
            tf.zeros([batch_size, hps.num_lstm_nodes[0]]),
            trainable=False
        )

        for i in range(num_timesteps):
            # [batch_size, 1, embed_size]
            embed_input = embed_inputs[:, i, :]
            embed_input = tf.reshape(embed_input,
                                     [batch_size, hps.num_embedding_size])
            forget_gate = tf.sigmoid(
                tf.matmul(embed_input, fx) + tf.matmul(h, fh) + fb)
            input_gate = tf.sigmoid(
                tf.matmul(embed_input, ix) + tf.matmul(h, ih) + ib)
            output_gate = tf.sigmoid(
                tf.matmul(embed_input, ox) + tf.matmul(h, oh) + ob)
            mid_state = tf.tanh(
                tf.matmul(embed_input, cx) + tf.matmul(h, ch) + cb)
            state = mid_state * input_gate + state * forget_gate
            h = output_gate * tf.tanh(state)
        last = h

    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(last,
                              hps.num_fc_nodes,
                              activation=tf.nn.relu,
                              name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout,
                                 num_classes,
                                 name='fc2')

    with tf.name_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=outputs)
        loss = tf.reduce_mean(softmax_loss)
        # [0, 1, 5, 4, 2] -> argmax: 2
        y_pred = tf.argmax(tf.nn.softmax(logits),
                           1,
                           output_type=tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        tvars = tf.trainable_variables()
        for var in tvars:
            tf.logging.info('variable name: %s' % (var.name))
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), hps.clip_lstm_grads)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step)

    return ((inputs, outputs, keep_prob),
            (loss, accuracy),
            (train_op, global_step))


placeholders, metrics, others = create_model(
    hps, vocab_size, num_classes)

inputs, outputs, keep_prob = placeholders
loss, accuracy = metrics
train_op, global_step = others

placeholders, metrics, others = create_model(
    hps, vocab_size, num_classes)

inputs, outputs, keep_prob = placeholders
loss, accuracy = metrics
train_op, global_step = others

placeholders, metrics, others = create_model(vocab_size, num_classes)
inputs, outputs, keep_prob = placeholders
loss, accuracy = metrics
train_op, global_step = others

init_op = tf.global_variables_initializer()
train_keep_prob_value = 0.8
test_keep_prob_value = 1.0

num_train_steps = 5000

# Train: 99.7%
# Valid: 92.7%
# Test:  93.2%
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(num_train_steps):
        batch_inputs, batch_labels = train_dataset.next_batch(
            hps.batch_size)
        outputs_val = sess.run([loss, accuracy, train_op, global_step],
                               feed_dict={
                                   inputs: batch_inputs,
                                   outputs: batch_labels,
                                   keep_prob: train_keep_prob_value,
                               })
        loss_val, accuracy_val, _, global_step_val = outputs_val
        if global_step_val % 20 == 0:
            tf.logging.info("Step: %5d, loss: %3.3f, accuracy: %3.3f"
                            % (global_step_val, loss_val, accuracy_val))
