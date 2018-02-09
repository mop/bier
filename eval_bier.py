import argparse
import numpy as np
import tensorflow as tf
import dataset
import train_bier as model
import code.deep_inception as models
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-labels', required=True)
    parser.add_argument('--test-images', required=True)
    parser.add_argument('--eval-every', type=int, default=None)
    parser.add_argument('--hidden-embedding-size', type=int, default=512)
    parser.add_argument('--embedding-sizes', type=str, default='96,160,256')
    parser.add_argument('--dump-only', action='store_true')
    parser.add_argument('--dump-prefix', type=str)
    parser.add_argument('--model', required=True)

    args = parser.parse_args()

    embedding_sizes = [int(x) for x in args.embedding_sizes.split(',')]
    spec = model.TrainingData(
        crop_size=224,
        channels=3,
        mean=(104.0, 117.0, 123.0))

    test_provider = dataset.NpyDatasetProvider(
        data_spec=spec,
        image_file=args.test_images,
        label_file=args.test_labels,
        batch_size=model.BATCH_SIZE,
        num_concurrent=1,
        is_training=False)

    test_labels, test_data = test_provider.dequeue_op
    test_net = models.GoogleNet({'data': test_data}, trainable=False)
    hidden_test_output = test_net.get_output()
    test_hidden_layer = test_net.get_output()
    test_preds, test_endpoints = model.embedding_tower(
        test_hidden_layer,
        embedding_sizes)

    fnames = list(glob.glob(os.path.join(args.model, '*meta')))
    checkpoints = [f.replace('.meta', '') for f in fnames]

    saver = tf.train.Saver()
    for checkpoint in checkpoints:
        if (args.eval_every is not None and
           int(checkpoint.split('-')[-1]) % args.eval_every != 0):
            continue
        with tf.Session() as sess:
            print(checkpoint)

            saver.restore(sess, checkpoint)
            test_provider.feed_data(sess)
            tf.train.start_queue_runners(sess)
            all_fvecs = []
            all_fvecs_hidden = []
            all_labels = []
            num_batches = int(np.ceil(
                test_provider.num_images / float(model.BATCH_SIZE)))
            print('Processing {} batches'.format(num_batches))
            for batch_idx in xrange(num_batches):
                fvec, fvec_hidden, cls = sess.run([
                    test_preds, hidden_test_output, test_labels])
                fvec = fvec[cls >= 0, ...]
                fvec_hidden = fvec_hidden[cls >= 0, ...]
                cls = cls[cls >= 0, ...]
                if args.dump_only:
                    all_fvecs.append(np.array(fvec, dtype=np.float16))
                    all_fvecs_hidden.append(np.array(fvec_hidden[:, 0, 0, :],
                                                     dtype=np.float16))
                    all_labels.append(np.array(cls))
                else:
                    all_fvecs.append(np.array(fvec))
                    all_fvecs_hidden.append(np.array(fvec_hidden[:, 0, 0, :]))
                    all_labels.append(np.array(cls))

            all_labels = np.concatenate(all_labels)

            if not args.dump_only:
                print('evaluating...')
                model.evaluate2(np.vstack(all_fvecs_hidden), all_labels)
                model.evaluate(np.vstack(all_fvecs), all_labels,
                               embedding_sizes)
            elif args.dump_prefix is not None:
                fname = args.dump_prefix + checkpoint.split('-')[-1] + '.npz'
                np.savez_compressed(
                    fname, fvecs_hidden=np.vstack(all_fvecs_hidden),
                    fvecs=np.vstack(all_fvecs), labels=all_labels)


if __name__ == '__main__':
    main()
