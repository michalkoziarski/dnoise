import os
import urllib
import tarfile
import zipfile
import cPickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import misc


class Image:
    def __init__(self, image=None, path=None, shape=None, keep_in_memory=True, preload=False, normalize=True,
                 noise=False, noise_mean=0.0, noise_std=0.1):
        if preload and not keep_in_memory:
            raise ValueError('Can\'t preload without keeping in memory')

        if image is None and path is None:
            raise ValueError('Needs either image or path')

        self.path = path
        self.shape = shape
        self.keep_in_memory = keep_in_memory
        self.preload = preload
        self.normalize = normalize
        self.noise = noise
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.image = None

        if preload or image is not None:
            self.load_and_process(image)

    def get(self):
        if self.image is not None:
            return self.image
        else:
            return self.load_and_process()

    def load_and_process(self, image=None):
        if image is None:
            image = misc.imread(self.path)

        if self.shape is not None:
            image = misc.imresize(image, self.shape)

        if self.normalize:
            image = image / 255.

        if self.noise:
            image += np.random.normal(self.noise_mean, self.noise_std, image.shape)

            upper_bound = 1.0 if self.normalize else 255

            image[image < 0] = 0
            image[image > upper_bound] = upper_bound

        if self.keep_in_memory:
            self.image = image

        return image

    def noisy(self, mean=0.0, std=0.1):
        return Image(image=self.image, path=self.path, shape=self.shape, keep_in_memory=True,
                     normalize=self.normalize, noise=True, noise_mean=mean, noise_std=std)

    def display(self, path=None, size=(128, 128)):
        image = self.get()

        if size is not None:
            image = misc.imresize(image, size)

        if path is None:
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        else:
            plt.imsave(path, image)


class Label:
    def __init__(self, label, one_hot=True, dictionary=None, length=None):
        if one_hot is True and dictionary is None and length is None:
            raise ValueError('If one_hot is true needs either dictionary or length')

        if one_hot:
            if dictionary is None:
                dictionary = range(length)

            if length is None:
                length = len(dictionary)

            self.label = np.zeros(length)
            self.label[dictionary.index(label)] = 1
        else:
            self.label = label

    def get(self):
        return self.label


class Batch:
    def __init__(self, images, targets, noise=False, noise_mean=0.0, noise_std=0.1):
        if noise:
            self._images = [image.noisy(noise_mean, noise_std) for image in images]
        else:
            self._images = images

        self._targets = targets
        self.noise = noise
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def images(self):
        return np.array([image.get() for image in self._images])

    def targets(self):
        return np.array([target.get() for target in self._targets])

    def noisy(self, mean=0.0, std=0.1):
        return Batch(images=self._images, targets=self._targets, noise=True, noise_mean=mean, noise_std=std)

    def size(self):
        return len(self._images)


class DataSet(Batch):
    def __init__(self, images, targets, batch_size=128):
        if len(images) != len(targets):
            raise ValueError('Images and targets should have the same size')

        self.batch_size = batch_size
        self.length = len(images)
        self.epochs_completed = 0
        self.current_index = 0

        Batch.__init__(self, images, targets)

    def batch(self, size=None):
        if size is None:
            size = self.batch_size

        batch_images = self._images[self.current_index:(self.current_index + size)]
        batch_targets = self._targets[self.current_index:(self.current_index + size)]

        self.current_index += size

        if self.current_index >= self.length:
            self.current_index = 0
            self.epochs_completed += 1

            perm = np.random.permutation(self.length)

            self._images = self._images[perm]
            self._targets = self._targets[perm]

        return Batch(batch_images, batch_targets)


class DataSets:
    def __init__(self, images, targets, batch_size=128, split=(0.6, 0.2, 0.2)):
        if sum(split) != 1.0:
            raise ValueError('Values of split should sum up to 1.0')

        if len(images) != len(targets):
            raise ValueError('Images and targets should have the same size')

        self.length = len(images)

        train_len = int(self.length * split[0])
        valid_len = int(self.length * split[1])

        idxs = range(self.length)

        train_idxs = np.random.choice(idxs, train_len)
        idxs = [idx for idx in idxs if idx not in train_idxs]
        valid_idxs = np.random.choice(idxs, valid_len)
        test_idxs = [idx for idx in idxs if idx not in valid_idxs]
        np.random.shuffle(test_idxs)

        train_images = np.array(images)[train_idxs]
        train_targets = np.array(targets)[train_idxs]
        valid_images = np.array(images)[valid_idxs]
        valid_targets = np.array(targets)[valid_idxs]
        test_images = np.array(images)[test_idxs]
        test_targets = np.array(targets)[test_idxs]

        self.train = DataSet(train_images, train_targets, batch_size)
        self.valid = DataSet(valid_images, valid_targets, batch_size)
        self.test = DataSet(test_images, test_targets, batch_size)


def load_face_image(batch_size=128, split=(0.6, 0.2, 0.2), shape=(64, 64), keep_in_memory=True, preload=False):
    root_path = '../data'
    data_path = os.path.join(root_path, 'FaceImage')
    tar_path = '%s.tar.gz' % data_path
    url = 'https://s3.amazonaws.com/michalkoziarski/FaceImage.tar.gz'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(data_path):
        if not os.path.exists(tar_path):
            urllib.urlretrieve(url, tar_path)

        with tarfile.open(tar_path) as tar:
            tar.extractall(root_path)

    genders = ['m', 'f']
    ages = ['(0, 2)', '(4, 6)', '(8, 13)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    dictionary = []

    for g in genders:
        for a in ages:
            dictionary.append('%s_%s' % (g, a))

    dfs = []

    for i in range(5):
        path = os.path.join(data_path, 'fold_%d_data.txt' % i)
        dfs.append(pd.read_csv(path, sep='\t'))

    df = pd.concat(dfs, ignore_index=True)
    df['path'] = df['user_id'] + '/landmark_aligned_face.' + df['face_id'].astype(str) + '.' + df['original_image']
    df['path'] = df['path'].apply(lambda x: os.path.join(data_path, 'aligned', x))
    df['age'] = df['age'].map(lambda x: x if x in ages else None)
    df['gender'] = df['gender'].map(lambda x: x if x in genders else None)
    df = df[['path', 'age', 'gender']].dropna()
    df['label'] = df['gender'].astype(str) + '_' + df['age'].astype(str)

    images = []
    targets = []

    for _, row in df.iterrows():
        path, _, _, label = row
        images.append(Image(path=path, shape=shape, keep_in_memory=keep_in_memory, preload=preload))
        targets.append(Label(label, dictionary=dictionary))

    return DataSets(images, targets, batch_size, split)


def load_mnist(batch_size=128, split=(0.6, 0.2, 0.2)):
    root_path = '../data'
    csv_path = os.path.join(root_path, 'mnist.csv')
    url = 'https://s3.amazonaws.com/michalkoziarski/mnist.csv'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(csv_path):
        urllib.urlretrieve(url, csv_path)

    images = []
    targets = []

    matrix = pd.read_csv(csv_path).as_matrix()

    for row in matrix:
        images.append(Image(image=np.reshape(row[1:], (28, 28))))
        targets.append(Label(row[0], length=10))

    return DataSets(images, targets, batch_size, split)


def load_cifar(batch_size=128, split=(0.6, 0.2, 0.2)):
    root_path = '../data'
    data_path = os.path.join(root_path, 'cifar-10-batches-py')
    tar_path = os.path.join(root_path, 'cifar-10-python.tar.gz')
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(data_path):
        if not os.path.exists(tar_path):
            urllib.urlretrieve(url, tar_path)

        with tarfile.open(tar_path) as tar:
            tar.extractall(root_path)

    images = []
    targets = []

    files = ['data_batch_%d' % i for i in range(1, 6)] + ['test_batch']
    paths = map(lambda x: os.path.join(data_path, x), files)

    for path in paths:
        with open(path, 'rb') as f:
            dict = cPickle.load(f)

        for i in range(len(dict['labels'])):
            image = np.reshape(dict['data'][i], (3, 32, 32)).transpose(1, 2, 0)
            images.append(Image(image=image))
            targets.append(Label(dict['labels'][i], length=10))

    return DataSets(images, targets, batch_size, split)


def load_gtsrb(batch_size=128, split=(0.6, 0.2, 0.2), shape=(32, 32), keep_in_memory=True, preload=False):
    root_path = '../data'
    data_path = os.path.join(root_path, 'GTSRB')
    img_path = os.path.join(data_path, 'Final_Training', 'Images')
    zip_path = os.path.join(root_path, 'GTSRB_Final_Training_Images.zip')
    url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(data_path):
        if not os.path.exists(zip_path):
            urllib.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root_path)

    images = []
    targets = []

    class_dirs = [o for o in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, o))]

    for class_dir in class_dirs:
        label = int(class_dir)
        class_path = os.path.join(img_path, class_dir)
        paths = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.ppm')]

        for path in paths:
            images.append(Image(path=path, shape=shape, keep_in_memory=keep_in_memory, preload=preload))
            targets.append(Label(label, length=43))

    return DataSets(images, targets, batch_size, split)
