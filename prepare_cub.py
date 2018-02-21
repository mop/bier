"""
This script prepares the CUB-200-2011 dataset for BIER.
We assume that the CUB images are stored in the images/ subdirectory.
"""
import cv2
import numpy as np
import os

TARGET_SIZE = 256


def prep_image(I):
    """
    Pads resizes the image to TARGET_SIZE x TARGET_SIZE by preserving the
    aspect ratio and padding it with a white border.

    Args:
        I: an input image

    Returns:
        A preprocessed image.
    """
    # height bigger than width
    if I.shape[0] >= I.shape[1]:
        scale = float(TARGET_SIZE) / I.shape[0]
        target_width = int(np.round(I.shape[1] * scale))
        I = cv2.resize(I, (target_width, TARGET_SIZE))

        # pad it...
        to_pad_left = (TARGET_SIZE - target_width) / 2
        to_pad_right = TARGET_SIZE - target_width - to_pad_left

        left = (np.ones((TARGET_SIZE, to_pad_left, 3)) * 255).astype(
            np.uint8)
        right = (np.ones((TARGET_SIZE, to_pad_right, 3)) * 255).astype(
            np.uint8)
        I = np.concatenate([left, I, right], axis=1)
    else:
        scale = float(TARGET_SIZE) / I.shape[1]
        target_height = int(np.round(I.shape[0] * scale))
        I = cv2.resize(I, (TARGET_SIZE, target_height))

        # pad it...
        to_pad_top = (TARGET_SIZE - target_height) / 2
        to_pad_bottom = TARGET_SIZE - target_height - to_pad_top

        top = (np.ones((to_pad_top, TARGET_SIZE, 3)) * 255).astype(np.uint8)
        bottom = (np.ones((to_pad_bottom, TARGET_SIZE, 3)) * 255).astype(
            np.uint8)
        I = np.concatenate([top, I, bottom], axis=0)

    return I


def collect_data(dirname):
    """
    Collects all images from the given directory.

    Args:
        dirname: The directory from which the images should be read.

    Returns:
        A list of preprocessed images
    """
    files = [os.path.join(dirname, f) for f in sorted(os.listdir(dirname))]

    all_images = []
    for f in files:
        I = cv2.imread(f)
        result_img = prep_image(I)
        all_images.append(result_img)
    return all_images


def main():
    dirs = sorted(os.listdir('images'))
    num_classes = len(dirs)

    train = dirs[:num_classes/2]
    test = dirs[num_classes/2:]

    print(train)
    print(test)

    train = [os.path.join('images', t) for t in train]
    test = [os.path.join('images', t) for t in test]

    all_train_images = []
    all_train_labels = []
    for label, t in enumerate(train):
        # bring into bc01 format and concatenate along axis 0
        imgs = np.concatenate([
            np.transpose(I, [2, 0, 1])[np.newaxis, ...]
            for I in collect_data(t)], axis=0)

        all_train_images.append(imgs)
        all_train_labels.append(np.ones(len(imgs),) * label)
    all_train_images = np.concatenate(all_train_images, axis=0)
    all_train_labels = np.concatenate(all_train_labels,
                                      axis=0).astype(np.int32)

    all_test_images = []
    all_test_labels = []
    for label, t in enumerate(test):
        # bring into bc01 format and concatenate along axis 0
        label = label + len(train)
        imgs = np.concatenate([np.transpose(I, [2, 0, 1])[np.newaxis, ...]
                               for I in collect_data(t)], axis=0)

        all_test_images.append(imgs)
        all_test_labels.append(np.ones(len(imgs),) * label)
    all_test_images = np.concatenate(all_test_images, axis=0)
    all_test_labels = np.concatenate(all_test_labels, axis=0).astype(np.int32)

    np.save('train_images.npy', all_train_images)
    np.save('train_labels.npy', all_train_labels)

    np.save('test_images.npy', all_test_images)
    np.save('test_labels.npy', all_test_labels)


if __name__ == '__main__':
    main()
