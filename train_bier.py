from __future__ import print_function
from __future__ import division

import argparse
import dataset
import collections
import numpy as np
import random
import flip_gradient

import code.deep_inception as models

import tensorflow as tf
from tensorflow.contrib import slim

TrainingData = collections.namedtuple(
    'TrainingData', ('crop_size', 'channels', 'mean'))

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0002
BATCH_SIZE = 32
LR_DECAY = 0.1
NUM_HIDDENS_ADVERSARIAL = 2
HIDDEN_ADVERSARIAL_SIZE = 512

LAMBDA_WEIGHT = 100.0

REGULARIZATION_CHOICES = ['activation', 'adversarial']

EMBEDDING_SCOPE_NAME = 'embedding_tower'
VERBOSE = False


def do_print(*args, **kwargs):
    """
    Wrapper around tf.Print to enable/disable verbose printing.
    """
    if VERBOSE:
        return tf.Print(*args, **kwargs)
    else:
        return args[0]


def embedding_tower(hidden_layer, embedding_sizes, reuse=False):
    """
    Creates the embedding tower on top of a feature extractor.

    Args:
        hidden_layer: Last hidden layer of the feature extractor.
        embedding_sizes: array indicating the sizes of our
                         embedding, e.g. [96, 160, 256]
        reuse: tensorflow reuse flag for the variable scope.
    Returns: A tuple consisting of the embedding and end_points.
    """
    end_points = {}
    final_layers = []

    with tf.variable_scope(EMBEDDING_SCOPE_NAME, reuse=reuse) as scope:
        hidden_layer = slim.flatten(hidden_layer)
        for idx, embedding_size in enumerate(embedding_sizes):

            scope_name = 'embedding/fc_{}'.format(idx)
            embedding = slim.fully_connected(
                hidden_layer, embedding_size, activation_fn=None,
                scope=scope_name)
            regul_out = slim.fully_connected(tf.stop_gradient(
                hidden_layer), embedding_size, scope=scope_name,
                reuse=True, activation_fn=None, biases_initializer=None)

            end_points['{}/embedding/fc_{}'.format(
                EMBEDDING_SCOPE_NAME, idx)] = embedding
            end_points['{}/embedding/fc_{}_regularizer'.format(
                EMBEDDING_SCOPE_NAME, idx)] = regul_out
            final_layers.append(embedding)

        embedding = tf.concat(final_layers, axis=1)
        end_points['{}/embedding'.format(EMBEDDING_SCOPE_NAME)] = embedding

        weight_variables = slim.get_variables_by_name('weights', scope=scope)
    for w in weight_variables:
        tf.add_to_collection('weights', w)
    return embedding, end_points


def evaluate2(fvecs, labels, tag='Hidden'):
    """
    Evaluation of a single embedding.

    Args:
        fvecs: numpy array of feature vectors
        labels: labels
    Returns:
        The recall @1
    """
    fvecs /= np.maximum(1e-5, np.linalg.norm(fvecs, axis=1, keepdims=True))
    D = fvecs.dot(fvecs.T)
    # Remove the diagonal for evalution! This is the same sample as the query.
    I = np.eye(D.shape[0]) * abs(D).max() * 10.0
    D -= I
    predictions = D.argmax(axis=1)
    pred_labels = labels[predictions]

    recall = (pred_labels == labels).sum() / float(len(labels))
    print('R@1 ({}): '.format(tag), recall)
    return recall


def evaluate(fvecs, labels, embedding_sizes):
    """
    Evaluation of a bier embedding.

    Args:
        fvecs: numpy array of feature vectors
        labels: labels
    Returns:
        The recall @1
    """
    embedding_scales = [float(e) / sum(embedding_sizes)
                        for e in embedding_sizes]
    start_idx = 0
    for e, s in zip(embedding_sizes, embedding_scales):
        stop_idx = start_idx + e
        evaluate2(np.array(fvecs[:, start_idx:stop_idx].copy(
        )), labels, tag='Embedding-{}'.format(e))
        fvecs[:, start_idx:stop_idx] /= np.maximum(1e-5, np.linalg.norm(
            fvecs[:, start_idx:stop_idx], axis=1, keepdims=True)) / s
        start_idx = stop_idx

    # Compute distance matrix.
    D = fvecs.dot(fvecs.T)
    I = np.eye(D.shape[0]) * abs(D).max() * 10.0
    D -= I

    # compute R@1
    predictions = D.argmax(axis=1)
    pred_labels = labels[predictions]

    recall = (pred_labels == labels).sum() / float(len(labels))
    print('R@1 (Embedding): ', (pred_labels == labels).sum() /
          float(len(labels)))
    return recall


def build_train(predictions, end_points, y, embedding_sizes,
                shrinkage=0.06,
                lambda_div=0.0, C=25, alpha=2.0, beta=0.5, initial_acts=0.5,
                eta_style=False, dtype=tf.float32, regularization=None):
    """
    Builds the boosting based training.

    Args:
        predictions: tensor of the embedding predictions
        end_points: dictionary of endpoints of the embedding tower
        y: tensor class labels
        embedding_sizes: list, which indicates the size of the sub-embedding
                         (e.g. [96, 160, 256])
        shrinkage: if you use eta_style = True, set to 1.0, otherwise keep it
                   small (e.g. 0.06).
        lambda_div: regularization parameter.
        C: parameter for binomial deviance.
        alpha: parameter for binomial deviance.
        dtype: data type for computations, typically tf.float32
        initial_acts: 0.5 if eta_style is false, 0.0 if eta_style is true
        regularization: regularization method (either activation or
                        adversarial)
    Returns:
        The training loss.
    """
    shape = predictions.get_shape().as_list()
    num_learners = len(embedding_sizes)
    # Pairwise labels.
    pairs = tf.reshape(
        tf.cast(tf.equal(y[:, tf.newaxis], y[tf.newaxis, :]), dtype), [-1])

    m = 1.0 * pairs + (-C * (1.0 - pairs))
    W = tf.reshape((1.0 - tf.eye(shape[0], dtype=dtype)), [-1])
    W = W * pairs / tf.reduce_sum(pairs) + W * \
        (1.0 - pairs) / tf.reduce_sum(1.0 - pairs)

    # * boosting_weights_init
    boosting_weights = tf.ones(shape=(shape[0] * shape[0],), dtype=dtype)

    normed_fvecs = []
    regular_fvecs = []

    # L2 normalize fvecs
    for i in xrange(len(embedding_sizes)):
        start = int(sum(embedding_sizes[:i]))
        stop = int(start + embedding_sizes[i])

        fvec = tf.cast(predictions[:, start:stop], dtype)
        regular_fvecs.append(fvec)
        fvec = do_print(fvec, [tf.norm(fvec, axis=1)],
                        'fvecs_{}_norms'.format(i))
        tf.summary.histogram('fvecs_{}'.format(i), fvec)
        tf.summary.histogram('fvecs_{}_norm'.format(i), tf.norm(fvec, axis=1))
        normed_fvecs.append(
            fvec / tf.maximum(tf.constant(1e-5, dtype=dtype),
                              tf.norm(fvec, axis=1, keep_dims=True)))

    alpha = tf.constant(alpha, dtype=dtype)
    beta = tf.constant(beta, dtype=dtype)
    C = tf.constant(C, dtype=dtype)
    shrinkage = tf.constant(shrinkage, dtype=dtype)

    loss = tf.constant(0.0, dtype=dtype)
    acts = tf.constant(initial_acts, dtype=dtype)
    tf.summary.histogram('boosting_weights_0', boosting_weights)
    tf.summary.histogram('boosting_weights_0_pos', tf.boolean_mask(
        boosting_weights, tf.equal(pairs, 1.0)))
    tf.summary.histogram('boosting_weights_0_neg', tf.boolean_mask(
        boosting_weights, tf.equal(pairs, 0.0)))
    Ds = []
    for i in xrange(len(embedding_sizes)):
        fvec = normed_fvecs[i]
        Ds.append(tf.matmul(fvec, tf.transpose(fvec)))

        D = tf.reshape(Ds[-1], [-1])
        my_act = alpha * (D - beta) * m
        my_loss = tf.log(tf.exp(-my_act) + tf.constant(1.0, dtype=dtype))
        tmp = (tf.reduce_sum(my_loss * boosting_weights * W) /
               tf.constant(num_learners, dtype=dtype))
        loss += tmp

        tf.summary.scalar('learner_loss_{}'.format(i), tmp)

        if eta_style:
            nu = 2.0 / (1.0 + 1.0 + i)
            if shrinkage != 1.0:
                acts = (1.0 - nu) * acts + nu * shrinkage * D
                inputs = alpha * (acts - beta) * m
                booster_loss = tf.log(tf.exp(-(inputs)) + 1.0)
                boosting_weights = tf.stop_gradient(
                    -tf.gradients(tf.reduce_sum(booster_loss), inputs)[0])
            else:
                acts = (1.0 - nu) * acts + nu * shrinkage * my_act
                booster_loss = tf.log(tf.exp(-acts) + 1.0)
                boosting_weights = tf.stop_gradient(
                    -tf.gradients(tf.reduce_sum(booster_loss), acts)[0])
        else:
            # simpler variant of the boosting algorithm.
            acts += shrinkage * (D - beta) * alpha * m
            booster_loss = tf.log(tf.exp(-acts) + 1.0)
            cls_weight = tf.cast(1.0 * pairs + (1.0 - pairs) * 2.0,
                                 dtype=dtype)
            boosting_weights = tf.stop_gradient(-tf.gradients(
                tf.reduce_sum(booster_loss), acts)[0] * cls_weight)

            tf.summary.histogram(
                'boosting_weights_{}'.format(i + 1), boosting_weights)
            pos_weights = tf.boolean_mask(
                boosting_weights, tf.equal(pairs, 1.0))
            neg_weights = tf.boolean_mask(
                boosting_weights, tf.equal(pairs, 0.0))
            pos_bins = tf.histogram_fixed_width(pos_weights, (tf.constant(
                0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)), nbins=10)
            neg_bins = tf.histogram_fixed_width(neg_weights, (tf.constant(
                0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)), nbins=10)
            loss = do_print(loss, [tf.reduce_mean(
                booster_loss)], 'Booster loss {}'.format(i + 1))
            loss = do_print(loss, [pos_bins, neg_bins],
                            'Positive and negative boosting weights {}'.format(
                                i + 1), summarize=100)

            tf.summary.histogram(
                'boosting_weights_{}_pos'.format(i + 1), pos_weights)
            tf.summary.histogram(
                'boosting_weights_{}_neg'.format(i + 1), neg_weights)
            tf.summary.scalar('booster_loss_{}'.format(
                i + 1), tf.reduce_mean(booster_loss))

    # add the independence loss
    tf.summary.scalar('discriminative_loss', loss)

    embedding_weights = [v for v in tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES) if 'embedding' in v.name and
                                          'weight' in v.name]
    if lambda_div > 0.0:
        loss += REGULARIZATION_FUNCTIONS[regularization](
            fvecs=normed_fvecs, end_points=end_points,
            embedding_weights=embedding_weights,
            embedding_sizes=embedding_sizes,
            lambda_weight=LAMBDA_WEIGHT) * lambda_div
    tf.summary.scalar('loss', loss)
    return loss


def build_pairwise_tower_loss(fvecs_i, fvecs_j, scope=None,
                              lambda_weight=LAMBDA_WEIGHT):
    """
    Builds an adversarial regressor from fvecs_j to fvecs_i.

    Args:
        fvecs_i: the target embedding (i.e. the smaller embedding)
        fvecs_j: the source embedding (i.e. the begger embedding)
        scope: scope name of the regressor.
        lambda_weight: the regularization parameter for the weights.
    Returns:
        An adversarial regressor loss function.
    """
    # build a regressor from fvecs_j to fvecs_i
    fvecs_i = flip_gradient.flip_gradient(fvecs_i)
    fvecs_j = flip_gradient.flip_gradient(fvecs_j)
    net = fvecs_j

    bias_loss = 0.0
    weight_loss = 0.0
    adversarial_loss = 0.0
    with tf.variable_scope(scope):
        for i in xrange(NUM_HIDDENS_ADVERSARIAL):
            if i < NUM_HIDDENS_ADVERSARIAL - 1:
                net = slim.fully_connected(
                    net, HIDDEN_ADVERSARIAL_SIZE, scope='fc_{}'.format(i),
                    activation_fn=tf.nn.relu)
            else:
                net = slim.fully_connected(net, fvecs_i.get_shape().as_list(
                )[-1], scope='fc_{}'.format(i), activation_fn=None)
            b = slim.get_variables(
                scope=tf.get_variable_scope().name + '/fc_{}/biases'.format(i)
            )[0]
            W = slim.get_variables(
                scope=tf.get_variable_scope().name + '/fc_{}/weights'.format(i)
            )[0]
            weight_loss += tf.reduce_mean(
                tf.square(tf.reduce_sum(W * W, axis=1) - 1)) * lambda_weight
            if b is not None:
                bias_loss += tf.maximum(
                    0.0,
                    tf.reduce_sum(b * b) - 1.0) * lambda_weight
        adversarial_loss += -tf.reduce_mean(tf.square(fvecs_i * net))
        tf.summary.scalar('adversarial loss', adversarial_loss)
        tf.summary.scalar('weight loss', weight_loss)
        tf.summary.scalar('bias loss', bias_loss)
    return adversarial_loss + weight_loss + bias_loss


def adversarial_loss(fvecs, end_points, embedding_weights, embedding_sizes,
                     lambda_weight=LAMBDA_WEIGHT):
    """
    Applies the adversarial loss on our embedding.

    Args:
        fvecs: tensor of the embedding feature vectors.
        end_points: dictionary of end_points of the embedding tower.
        embedding_weights: weight matrices of the embedding.
        embedding_sizes: list of embedding sizes, e.g. [96, 160, 256]
        lambda_weight: weight regularization parameter.
    Returns:
        The regularization loss.
    """
    loss = 0.0
    with tf.variable_scope('pws'):
        for layer_idx, fvecs in enumerate(iterate_regularization_acts(
                end_points, embedding_sizes)):

            for i in xrange(len(fvecs)):
                for j in xrange(i + 1, len(fvecs)):
                    name = 'pw_tower_loss_layer_{}_from_{}_to_{}'.format(
                        layer_idx, i, j)

                    loss += build_pairwise_tower_loss(
                        fvecs[i], fvecs[j],
                        name,
                        lambda_weight=lambda_weight)

    weight_loss = 0.0
    for W in embedding_weights:
        weight_loss += tf.reduce_mean(
            tf.square(tf.reduce_sum(W * W, axis=1) - 1))
    weight_loss = do_print(weight_loss, [weight_loss], 'weight loss')
    loss = do_print(loss, [loss], 'adversarial correlation dann hidden loss')
    tf.summary.scalar('adversarial correlation dann hidden losss', loss)
    tf.summary.scalar('weight loss', weight_loss)

    return loss + lambda_weight * weight_loss


def iterate_regularization_acts(end_points, embedding_sizes):
    """
    Iterates through the regularization activations.

    Args:
        end_points: Dictionary of end_points.
        embedding_sizes: List of embedding sizes, e.g. [96, 160, 256].
    Yields:
        All iteration endpoints
    """
    num_embeddings = len(embedding_sizes)

    fvecs = []
    # yield the output layer.
    for i in xrange(num_embeddings):
        fvecs.append(end_points[EMBEDDING_SCOPE_NAME +
                                '/embedding/fc_{}_regularizer'.format(i)])
    yield fvecs


def activation_loss(fvecs, end_points, embedding_weights, embedding_sizes,
                    lambda_weight=LAMBDA_WEIGHT):
    """
    Applies the activation loss on our embedding.

    Args:
        fvecs: embedding tensors.
        end_points: dictionary of end_points from embedding_tower.
        embedding_weights: weight matrices of embeddings
        embedding_sizes: list of embedding sizes, e.g. [96, 160, 256].
        lambda_weight: Weight regularization parameter.
    Returns:
        The activation loss.
    """
    loss = 0.0
    for fvecs in iterate_regularization_acts(end_points, embedding_sizes):
        print(fvecs)
        for i in xrange(len(fvecs)):
            for j in xrange(i + 1, len(fvecs)):
                loss += tf.reduce_mean(
                    tf.square(fvecs[i][:, tf.newaxis, :] *
                              fvecs[j][:, :, tf.newaxis]))

    weight_loss = 0.0
    for W in embedding_weights:
        weight_loss += tf.reduce_mean(
            tf.square(tf.reduce_sum(W * W, axis=0) - 1))
    weight_loss = do_print(weight_loss, [weight_loss], 'weight loss')
    loss = do_print(loss, [loss], 'group loss')

    return loss + weight_loss * lambda_weight


def main():
    global NUM_HIDDENS_ADVERSARIAL
    global HIDDEN_ADVERSARIAL_SIZE
    global BATCH_SIZE
    global LR_DECAY
    global LAMBDA_WEIGHT

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-images', required=True)
    parser.add_argument('--train-labels', required=True)
    parser.add_argument('--test-images', required=False)
    parser.add_argument('--test-labels', required=False)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--weights', default='data/inception.npy')
    parser.add_argument('--lambda-weight', type=float, default=LAMBDA_WEIGHT)
    parser.add_argument('--lambda-div', type=float, default=0.0)
    parser.add_argument('--shrinkage', type=float, default=0.06)
    parser.add_argument('--eta-style', action='store_true')
    parser.add_argument('--lr-decay', type=float, default=LR_DECAY)
    parser.add_argument('--embedding-sizes', type=str, default='96,160,256')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--num-iterations', type=int, default=20000)
    parser.add_argument('--logdir', type=str, default='train')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--regularization', type=str, default='activation',
                        choices=REGULARIZATION_CHOICES)
    parser.add_argument('--hidden-adversarial-size',
                        type=int, default=HIDDEN_ADVERSARIAL_SIZE)
    parser.add_argument('--num-hidden-adversarial', type=int,
                        default=NUM_HIDDENS_ADVERSARIAL)
    parser.add_argument('--labels-per-batch', type=int, default=6)
    parser.add_argument('--images-per-identity', type=int)
    parser.add_argument('--embedding-lr-multiplier', type=float, default=10.0)
    parser.add_argument('--lr-anneal', type=int)
    parser.add_argument('--use-same-learnrate', action='store_true')
    parser.add_argument('--skip-test', action='store_true')

    dtype = tf.float32

    args = parser.parse_args()
    LAMBDA_WEIGHT = args.lambda_weight
    LR_DECAY = args.lr_decay
    BATCH_SIZE = args.batch_size
    NUM_HIDDENS_ADVERSARIAL = args.num_hidden_adversarial
    HIDDEN_ADVERSARIAL_SIZE = args.hidden_adversarial_size
    print(args.logdir)

    skip_test = args.skip_test
    if args.test_images is None or args.test_labels is None:
        skip_test = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    embedding_sizes = [int(x) for x in args.embedding_sizes.split(',')]

    spec = TrainingData(crop_size=224, channels=3, mean=(
        104.0, 117.0, 123.0))

    print('creating datasets...')
    train_provider = dataset.NpyDatasetProvider(
        data_spec=spec,
        labels_per_batch=args.labels_per_batch,
        images_per_identity=args.images_per_identity,
        image_file=args.train_images,
        label_file=args.train_labels,
        batch_size=BATCH_SIZE)
    test_provider = None
    train_labels, train_data = train_provider.dequeue_op
    if not skip_test:
        test_provider = dataset.NpyDatasetProvider(
            data_spec=spec,
            image_file=args.test_images,
            label_file=args.test_labels,
            batch_size=BATCH_SIZE,
            is_training=False)
        test_labels, test_data = test_provider.dequeue_op

    net = models.GoogleNet({'data': train_data})
    hidden_layer = net.get_output()
    preds, end_points = embedding_tower(
        hidden_layer, embedding_sizes)
    end_points['pool5_7x7_s1'] = hidden_layer

    if not skip_test:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            test_net = models.GoogleNet({'data': test_data}, trainable=False)
            test_hidden_layer = test_net.get_output()
            test_preds, test_endpoints = embedding_tower(
                test_hidden_layer,
                embedding_sizes,
                reuse=True)

    loss = build_train(
        preds,
        end_points,
        train_labels,
        embedding_sizes,
        shrinkage=args.shrinkage,
        lambda_div=args.lambda_div,
        eta_style=args.eta_style,
        dtype=dtype,
        regularization=args.regularization)

    # Add weight decay.
    all_weights = tf.get_collection('weights')
    all_weights = list(set(all_weights))
    for w in all_weights:
        if 'embedding' not in w.name:
            loss += tf.cast(tf.reduce_sum(w * w) * WEIGHT_DECAY, dtype=dtype)

    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    hidden_vars = [v for v in all_vars if 'embedding' not in v.name]
    embedding_vars = [v for v in all_vars if 'embedding' in v.name]

    global_step = tf.train.get_or_create_global_step()

    lr = tf.constant(LEARNING_RATE, dtype=tf.float32,
                     shape=(), name='learning_rate')
    if args.lr_anneal:
        lr = tf.train.exponential_decay(
            lr, global_step, args.lr_anneal, LR_DECAY, staircase=True)
    lr = do_print(lr, [lr], 'learning rate')

    opt_hidden = tf.train.AdamOptimizer(learning_rate=lr)
    train_op_hidden = opt_hidden.minimize(loss, var_list=hidden_vars)

    opt_embedding = tf.train.AdamOptimizer(
        learning_rate=lr * args.embedding_lr_multiplier)
    train_op_embedding = opt_embedding.minimize(
        loss, global_step=global_step, var_list=embedding_vars)

    with tf.control_dependencies([train_op_hidden, train_op_embedding]):
        train_op = tf.no_op()

    init_op = tf.global_variables_initializer()

    with tf.control_dependencies([init_op]):
        load_train_op = net.create_load_op(args.weights, ignore_missing=True)
        if not skip_test:
            load_test_op = test_net.create_load_op(
                args.weights, ignore_missing=True)

    checkpoint_saver = tf.train.CheckpointSaverHook(
        args.logdir,
        save_steps=args.eval_every,
        saver=tf.train.Saver(max_to_keep=100000))
    latest_checkpoint = tf.train.latest_checkpoint(args.logdir)
    need_init = latest_checkpoint is None
    assign_op = None
    start_iter = 0
    if not need_init:
        start_iter = int(latest_checkpoint.split('-')[-1])
        assign_op = global_step.assign(start_iter)

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=args.logdir,
            is_chief=True,
            hooks=[checkpoint_saver],
            save_checkpoint_secs=None) as sess:

        if need_init:
            sess.run(init_op)
            sess.run(load_train_op)
            if not skip_test:
                sess.run(load_test_op)
        else:
            sess.run(assign_op)

        if not args.skip_test:
            hidden_test_output = test_net.get_output()

        writer = tf.summary.FileWriter(args.logdir)
        for i in xrange(start_iter, args.num_iterations):
            if not args.skip_test and i % args.eval_every == 0:
                test_provider.feed_data(sess)
                all_fvecs = []
                all_fvecs_hidden = []
                all_labels = []
                #all_caffe_fvecs = []
                num_batches = int(
                    np.ceil(test_provider.num_images / float(BATCH_SIZE)))
                print('Evaluating {} batches'.format(num_batches))
                for batch_idx in xrange(num_batches):
                    fvec, fvec_hidden, cls = sess.run(
                        [test_preds, hidden_test_output, test_labels])
                    fvec = fvec[cls >= 0, ...]
                    fvec_hidden = fvec_hidden[cls >= 0, ...]
                    cls = cls[cls >= 0, ...]
                    all_fvecs.append(np.array(fvec))
                    all_fvecs_hidden.append(np.array(fvec_hidden[:, 0, 0, :]))
                    all_labels.append(np.array(cls))

                all_labels = np.concatenate(all_labels)

                recall = evaluate2(np.vstack(all_fvecs_hidden), all_labels)
                summary = tf.Summary(value=[tf.Summary.Value(
                    tag='Recall@1_Hidden_Layer', simple_value=recall)])
                writer.add_summary(summary, i)

                recall = evaluate(np.vstack(all_fvecs),
                                  all_labels, embedding_sizes)
                summary = tf.Summary(value=[tf.Summary.Value(
                    tag='Recall@1_Embedding_Layer', simple_value=recall)])
                writer.add_summary(summary, i)

            lossval, _ = sess.run([loss, train_op])
            if i % 40 == 0:
                print('loss: {}@Iteration {}'.format(lossval, i))


REGULARIZATION_FUNCTIONS = {
    'activation': activation_loss,
    'adversarial': adversarial_loss,
}

if __name__ == '__main__':
    main()
