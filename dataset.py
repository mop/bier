from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def process_image(img, crop=None, mean=None,
                  mirror=True, is_training=True):
    """
    Preprocessing code. For training this function randomly crop images and
    flips the image randomly.
    For testing we use the center crop of the image.

    Args:
        img: The input image.
        crop: Size of the output image.
        mean: three dimensional array indicating the mean values to subtract
              from the image.
        mirror: Flag, which indicates if images should be mirrored.
        is_training: Flag which indicates whether training preprocessing
                     or testing preprocessing should be used.

    Returns:
        A pre-processed image.
    """
    if is_training:
        img = tf.random_crop(img, [crop, crop, img.get_shape().as_list()[
                             2]], name='random_image_crop')

        if mirror:
            img = tf.image.random_flip_left_right(img)
    else:
        new_shape = img.get_shape().as_list()[0]
        offset = (new_shape - crop) // 2
        img = tf.slice(img, begin=tf.convert_to_tensor(
            [offset, offset, 0]), size=tf.convert_to_tensor([crop, crop, -1]))

    # Mean subtraction
    return tf.to_float(img) - mean


class NpyDatasetProvider(object):
    """
    This class hooks up a numpy dataset file to tensorflow queue runners.
    """
    def __init__(self, data_spec, image_file, label_file,
                 labels_per_batch=6, images_per_identity=None,
                 batch_size=32, is_training=True, num_concurrent=4):
        super(NpyDatasetProvider, self).__init__()
        # The data specifications describe how to process the image
        self.data_spec = data_spec

        self.images = np.transpose(np.load(image_file), (0, 2, 3, 1))

        self.labels = np.load(label_file)
        if self.labels.dtype != np.int32:
            self.labels = self.labels.astype(np.int32)

        self.original_num_images = len(self.images)
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_images = len(self.images)
        self.images_per_identity = images_per_identity

        if images_per_identity is not None:
            batch_size = images_per_identity * labels_per_batch
            self.batch_size = batch_size

        if not self.is_training and self.num_images % self.batch_size != 0:
            to_pad = self.batch_size - (self.num_images % self.batch_size)
            pad_img = np.zeros(
                [to_pad] + list(self.images.shape[1:]), dtype=np.uint8)
            pad_label = -np.ones([to_pad], dtype=np.int32)
            self.labels = np.r_[self.labels, pad_label]
            self.images = np.vstack([self.images, pad_img])

            self.num_images = len(self.images)

        self.indices = np.arange(len(self.images))
        self.labels_per_batch = labels_per_batch
        self.unique_labels = np.unique(self.labels)
        self.tf_unique_labels = tf.convert_to_tensor(self.unique_labels)

        self.setup(num_concurrent)

    def _setup_test(self, num_concurrent):
        """
        Setup the test queue.

        Args:
            num_concurrent: Number of concurrent threads.
        """
        num_images = len(self.images)
        self.num_batches = num_images // self.batch_size
        indices = tf.range(num_images)

        self.preprocessing_queue = tf.FIFOQueue(capacity=len(self.images),
                                                dtypes=[tf.int32],
                                                shapes=[()],
                                                name='preprocessing_queue')
        self.test_queue_op = self.preprocessing_queue.enqueue_many([indices])

        image_shape = (self.data_spec.crop_size,
                       self.data_spec.crop_size, self.data_spec.channels)
        processed_queue = tf.FIFOQueue(capacity=len(self.images),
                                       dtypes=[tf.int32, tf.float32],
                                       shapes=[(), image_shape],
                                       name='processed_queue')
        label, img = self.process_test()
        enqueue_processed_op = processed_queue.enqueue([label, img])

        self.dequeue_op = processed_queue.dequeue_many(self.batch_size)
        num_concurrent = min(num_concurrent, num_images)
        self.queue_runner = tf.train.QueueRunner(
            processed_queue,
            [enqueue_processed_op] * num_concurrent)
        tf.train.add_queue_runner(self.queue_runner)

    def setup(self, num_concurrent):
        """
        Setup of the queues.

        Args:
            num_concurrent: Number of concurrent threads.
        """
        if self.is_training:
            return self._setup_train(num_concurrent)
        else:
            return self._setup_test(num_concurrent)

    def _setup_train(self, num_concurrent):
        """
        Setup of the training queues.

        Args:
            num_concurrent: Number of concurrent threads.
        """
        num_images = len(self.images)
        self.num_batches = num_images // self.batch_size

        # Crate a label queue.
        self.label_queue = tf.RandomShuffleQueue(
            capacity=len(self.unique_labels),
            min_after_dequeue=self.labels_per_batch,
            dtypes=[tf.int32],
            shapes=[()],
            name='label_queue')
        self.label_queue_op = self.label_queue.enqueue_many(
            [self.tf_unique_labels])

        (labels, processed_images) = self.process()
        # print(labels.get_shape().as_list())
        # print(processed_images.get_shape().as_list())

        image_shape = (self.data_spec.crop_size,
                       self.data_spec.crop_size, self.data_spec.channels)
        processed_queue = tf.FIFOQueue(  # capacity=len(self.images),
            capacity=self.batch_size * 6,
            dtypes=[tf.int32, tf.float32],
            shapes=[(), image_shape],
            name='processed_queue')

        # Enqueue the processed image and path
        enqueue_processed_op = processed_queue.enqueue_many(
            [labels, processed_images])

        self.dequeue_op = processed_queue.dequeue_many(self.batch_size)
        num_concurrent = min(num_concurrent, num_images)
        self.label_runner = tf.train.QueueRunner(
            self.label_queue, [self.label_queue_op] * (num_concurrent + 1))
        self.queue_runner = tf.train.QueueRunner(
            processed_queue,
            [enqueue_processed_op] * num_concurrent)
        tf.train.add_queue_runner(self.label_runner)
        tf.train.add_queue_runner(self.queue_runner)

    def start(self, session, coordinator, num_concurrent=4):
        """
        Start the processing worker threads.

        Args:
            session: A tensorflow session.
            coordinator: A tensorflow coordinator.
            num_concurrent: Number of concurrent threads.

        Returns:
            a create threads operation.
        """
        if self.is_training:
            self.label_runner.create_threads(
                session, coord=coordinator, start=True)
        else:
            session.run(self.test_queue_op)  # just enqueue labels once!
        return self.queue_runner.create_threads(
            session, coord=coordinator, start=True)

    def feed_data(self, session):
        """
        Call this function for testing NpyDatasetProvider. It pushes
        the testing dataset once into the queue.

        Args:
            session: A tensorflow session
        """
        assert(not self.is_training)
        session.run(self.test_queue_op)  # just enqueue labels once!

    def get_labels(self, session):
        """
        Returns a list of labels from the queue.

        Args:
            session: A tensorflow session.

        Returns:
            An array of labels from the queue.
        """
        labels = session.run(
            self.label_queue.dequeue_many(self.labels_per_batch))
        return labels

    def get(self, session):
        """
        Get a single batch of images along with their labels.

        Returns:
            a tuple of (labels, images)
        """
        (labels, images) = session.run(self.dequeue_op)
        return (labels, images)

    def batches(self, session):
        """
        Yield a batch until no more images are left.

        Yields:
            Tuples in the form (labels, images)
        """
        for _ in xrange(self.num_batches):
            yield self.get(session=session)

    def process_test(self):
        """
        Processes the test images.

        Returns:
            Tuple consisting of (label, processed_image).
        """
        def fetch_images(the_idx):
            return self.images[the_idx, ...]

        def fetch_labels(the_idx):
            return self.labels[the_idx]

        index = self.preprocessing_queue.dequeue()
        label = tf.py_func(fetch_labels, [index], tf.int32)
        label.set_shape([])
        the_img = tf.py_func(fetch_images, [index], tf.uint8)
        the_img.set_shape(self.images.shape[1:])
        processed_img = process_image(img=the_img,
                                      #img=self.tf_images[index, ...],
                                      crop=self.data_spec.crop_size,
                                      mean=self.data_spec.mean,
                                      is_training=self.is_training)
        # return (self.tf_labels[index], processed_img)
        return (label, processed_img)

    def process(self):
        """
        Processes a training image.

        Returns:
            A tuple consisting of (label, image).
        """
        # Dequeue a single image path
        def fetch_data(sampled_labels):
            if self.images_per_identity is not None:
                all_ids = []
                for label in sampled_labels:
                    valid_mask, = np.nonzero(self.labels == label)
                    try:
                        valid_mask = np.random.choice(
                            valid_mask, size=self.images_per_identity,
                            replace=False)
                    except:
                        valid_mask = np.random.choice(
                            valid_mask, size=self.images_per_identity,
                            replace=True)  # well, whatever...
                    all_ids.append(valid_mask)
                all_ids = np.concatenate(all_ids)
                valid_labels = self.labels[all_ids]
                valid_images = self.images[all_ids, ...]
                return valid_labels, valid_images

            else:
                valid_mask, = np.nonzero(np.in1d(self.labels, sampled_labels))
                valid_mask = np.random.choice(valid_mask, size=self.batch_size)
                valid_labels = self.labels[valid_mask]
                valid_images = self.images[valid_mask, ...]
                return valid_labels, valid_images

        labels = self.label_queue.dequeue_many(self.labels_per_batch)
        labels.set_shape([self.labels_per_batch])

        labels, images = tf.py_func(fetch_data, [labels], [tf.int32, tf.uint8])
        labels.set_shape([self.batch_size])
        images.set_shape([self.batch_size] + list(self.images.shape[1:]))

        processed_images = []
        for i in xrange(self.batch_size):
            # Process the image
            processed_img = process_image(img=images[i, ...],
                                          crop=self.data_spec.crop_size,
                                          mean=self.data_spec.mean,
                                          is_training=self.is_training)
            processed_images.append(processed_img)
        processed_images = tf.stack(processed_images)
        return (labels, processed_images)
