import os
import urllib
import tarfile
import numpy as np
import pandas as pd

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


def _imagenet_path(element=None):
    if element is not None:
        return os.path.join(ROOT_PATH, 'ImageNet', element)
    else:
        return os.path.join(ROOT_PATH, 'ImageNet')


def _load_imagenet_images(dataset, shape, grayscale, normalize=True, n=None):
    assert os.path.exists(_imagenet_path(dataset))

    result = []

    for (dirpath, _, filenames) in os.walk(_imagenet_path(dataset)):
        for filename in filenames:
            path = os.path.join(ROOT_PATH, 'ImageNet', dirpath, filename)
            result.append(Image(path=path, shape=shape, keep_in_memory=False, grayscale=grayscale, normalize=normalize))

            if n is not None and len(result) >= n:
                return result

    return result


def load_imagenet_labeled(batch_size=50, shape=None, grayscale=False, patch=None, normalize=True, offset=None,
                          train_noise=None, test_noise=None, noise_before_resize=True):
    assert os.path.exists(_imagenet_path())

    for f in ['synsets.csv', 'val_ground_truth.csv']:
        if not os.path.exists(_imagenet_path(f)):
            url = 'https://raw.githubusercontent.com/michalkoziarski/datasets/master/ImageNet/%s' % f
            urllib.urlretrieve(url, _imagenet_path(f))

    synsets = pd.read_csv(_imagenet_path('synsets.csv'))
    val_ground_truth = pd.read_csv(_imagenet_path('val_ground_truth.csv'))

    train_images = _load_imagenet_images('train', shape, grayscale, normalize=normalize)
    val_images = _load_imagenet_images('val', shape, grayscale, normalize=normalize)

    train_targets = []
    val_targets = []

    for image in train_images:
        wnid = os.path.split(image.path)[-1].split('_')[0]
        label = int(synsets[synsets['WNID'] == wnid]['LABEL'])
        train_targets.append(Label(label - 1, length=1000))

    for image in val_images:
        id = int(os.path.split(image.path)[-1].split('.')[0].split('_')[-1])
        label = int(val_ground_truth[val_ground_truth['ID'] == id]['LABEL'])
        val_targets.append(Label(label - 1, length=1000))

    train_set = LabeledDataSet(train_images, train_targets, patch=patch, batch_size=batch_size, noise=train_noise,
                               offset=offset, noise_before_resize=noise_before_resize)
    val_set = LabeledDataSet(val_images, val_targets, patch=patch, batch_size=batch_size, noise=test_noise,
                             offset=offset, noise_before_resize=noise_before_resize)

    return train_set, val_set


def load_imagenet_labeled_validation(batch_size=50, shape=None, grayscale=False, patch=None, normalize=True,
                                     offset=None, noise=None, noise_before_resize=True, network=None, n=None):
    assert os.path.exists(_imagenet_path())

    if not os.path.exists(_imagenet_path('val_ground_truth.csv')):
        url = 'https://raw.githubusercontent.com/michalkoziarski/datasets/master/ImageNet/val_ground_truth.csv'
        urllib.urlretrieve(url, _imagenet_path('val_ground_truth.csv'))

    val_ground_truth = pd.read_csv(_imagenet_path('val_ground_truth.csv'))

    val_images = _load_imagenet_images('val', shape, grayscale, normalize=normalize, n=n)
    val_targets = []

    for image in val_images:
        id = int(os.path.split(image.path)[-1].split('.')[0].split('_')[-1])
        label = int(val_ground_truth[val_ground_truth['ID'] == id]['LABEL'])
        val_targets.append(Label(label - 1, length=1000))

    val_set = LabeledDataSet(val_images, val_targets, patch=patch, batch_size=batch_size, noise=noise, offset=offset,
                             noise_before_resize=noise_before_resize, network=network)

    return val_set


def load_imagenet_unlabeled(batch_size=50, shape=None, grayscale=False, noise=None, patch=None, sample=None,
                            normalize=True, offset=None, noise_before_resize=True):
    train_images = _load_imagenet_images('train', shape, grayscale, normalize=normalize)
    val_images = _load_imagenet_images('val', shape, grayscale, normalize=normalize)

    train_set = UnlabeledDataSet(train_images, noise=noise, patch=patch, sample=sample, batch_size=batch_size,
                                 offset=offset, noise_before_resize=noise_before_resize)
    val_set = UnlabeledDataSet(val_images, noise=noise, patch=patch, sample=sample, batch_size=batch_size,
                               offset=offset, noise_before_resize=noise_before_resize)

    return train_set, val_set


def load_imagenet_unlabeled_validation(batch_size=50, shape=None, grayscale=False, noise=None, patch=None, sample=None,
                                       normalize=True, offset=None, noise_before_resize=True, shuffle=True, n=None):
    val_images = _load_imagenet_images('val', shape, grayscale, normalize=normalize, n=n)

    val_set = UnlabeledDataSet(val_images, noise=noise, patch=patch, sample=sample, batch_size=batch_size,
                               offset=offset, noise_before_resize=noise_before_resize, shuffle=shuffle)

    return val_set
