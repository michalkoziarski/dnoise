import os
import urllib
import tarfile
import numpy as np

from containers import Image, Label, LabeledDataSet, UnlabeledDataSet

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _download_stl():
    stl_path = os.path.join(ROOT_PATH, 'STL-10')
    data_path = os.path.join(stl_path, 'stl10_binary')
    tar_path = os.path.join(stl_path, 'stl10_binary.tar.gz')
    url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

    if not os.path.exists(ROOT_PATH):
        os.makedirs(ROOT_PATH)

    if not os.path.exists(stl_path):
        os.makedirs(stl_path)

    if not os.path.exists(data_path):
        if not os.path.exists(tar_path):
            urllib.urlretrieve(url, tar_path)

        with tarfile.open(tar_path) as tar:
            tar.extractall(stl_path)

    return data_path


def _load_stl_images(path, shape, grayscale):
    stl_path = os.path.join(ROOT_PATH, 'STL-10')
    data_path = os.path.join(stl_path, 'stl10_binary')

    result = []

    with open(os.path.join(data_path, path), 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)

        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))

    for image in images:
        result.append(Image(image=image, shape=shape, keep_in_memory=True, grayscale=grayscale))

    return result


def _load_stl_targets(path):
    stl_path = os.path.join(ROOT_PATH, 'STL-10')
    data_path = os.path.join(stl_path, 'stl10_binary')

    result = []

    with open(os.path.join(data_path, path), 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)

    for label in labels:
        result.append(Label(label - 1, length=10))

    return result


def load_stl_labeled(batch_size=50, shape=None, grayscale=False):
    _download_stl()

    train_images = _load_stl_images('train_X.bin', shape, grayscale)
    test_images = _load_stl_images('test_X.bin', shape, grayscale)
    train_targets = _load_stl_targets('train_y.bin')
    test_targets = _load_stl_targets('test_y.bin')

    train_set = LabeledDataSet(train_images, train_targets, batch_size)
    test_set = LabeledDataSet(test_images, test_targets, batch_size)

    return train_set, test_set


def load_stl_unlabeled(batch_size=50, shape=None, grayscale=False, noise=None, patch=None):
    _download_stl()

    train_images = _load_stl_images('unlabeled_X.bin', shape, grayscale)
    test_images = _load_stl_images('train_X.bin', shape, grayscale)

    train_set = UnlabeledDataSet(train_images, noise=noise, patch=patch, batch_size=batch_size)
    test_set = UnlabeledDataSet(test_images, patch=patch, batch_size=batch_size)

    return train_set, test_set


def _imagenet_path(dataset):
    return os.path.join(ROOT_PATH, 'ImageNet', dataset)


def _load_imagenet_images(dataset, shape, grayscale):
    assert os.path.exists(_imagenet_path(dataset))

    result = []

    for (dirpath, _, filenames) in os.walk(_imagenet_path(dataset)):
        for filename in filenames:
            path = os.path.join(ROOT_PATH, 'ImageNet', dirpath, filename)
            result.append(Image(path=path, shape=shape, keep_in_memory=False, grayscale=grayscale))

    return result


def load_imagenet_unlabeled(batch_size=50, shape=None, grayscale=False, noise=None, patch=None):
    train_images = _load_imagenet_images('train', shape, grayscale)
    val_images = _load_imagenet_images('val', shape, grayscale)
    test_images = _load_imagenet_images('test', shape, grayscale)

    train_set = UnlabeledDataSet(train_images, noise=noise, patch=patch, batch_size=batch_size)
    val_set = UnlabeledDataSet(val_images, noise=noise, patch=patch, batch_size=batch_size)
    test_set = UnlabeledDataSet(test_images, noise=noise, patch=patch, batch_size=batch_size)

    return train_set, val_set, test_set
