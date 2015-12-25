import os
import urllib
import tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import misc


class Image:
    def __init__(self, path, shape=(20, 20), keep_in_memory=True, preload=False):
        if preload and not keep_in_memory:
            raise ValueError('Can\'t preload without keeping in memory')

        self.path = path
        self.shape = shape
        self.keep_in_memory = keep_in_memory
        self._image = None

        if preload:
            self.get()

    def get(self):
        if self._image is not None:
            return self._image
        else:
            image = misc.imread(self.path)

            if self.shape is not None:
                image = misc.imresize(image, self.shape)

            if self.keep_in_memory:
                self._image = image

            return image

    def display(self):
        plt.imshow(self.get())
        plt.axis('off')
        plt.show()


class Batch:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self._tensor = None

    def tensor(self):
        if self._tensor is None:
            self._tensor = np.array([image.get() for image in self.images])

        return self._tensor


class DataSet(Batch):
    def __init__(self, images, labels, batch_size=128):
        if len(images) != len(labels):
            raise ValueError('Images and labels should have the same size')

        self.batch_size = batch_size
        self.length = len(images)
        self.epochs_completed = 0
        self.current_index = 0

        Batch.__init__(self, images, labels)

    def batch(self):
        batch_images = self.images[self.current_index:(self.current_index + self.batch_size)]
        batch_labels = self.labels[self.current_index:(self.current_index + self.batch_size)]

        self.current_index += self.batch_size

        if self.current_index >= self.length:
            self.current_index = 0
            self.epochs_completed += 1

            perm = np.random.permutation(self.length)

            self.images = self.images[perm]
            self.labels = self.labels[perm]

        return Batch(batch_images, batch_labels)
        

class DataSets:
    def __init__(self, images, labels, batch_size=128, split=[0.7, 0.0, 0.3]):
        if sum(split) != 1.0:
            raise ValueError('Values of split should sum up to 1.0')

        if len(images) != len(labels):
            raise ValueError('Images and labels should have the same size')

        self.length = len(images)
        
        train_len = int(self.length * split[0])
        valid_len = int(self.length * split[1])

        idxs = range(self.length)

        train_idxs = np.random.choice(idxs, train_len)
        idxs = [idx for idx in idxs if idx not in train_idxs]
        valid_idxs = np.random.choice(idxs, valid_len)
        test_idxs = [idx for idx in idxs if idx not in valid_idxs]

        train_images = np.array(images)[train_idxs]
        train_labels = np.array(labels)[train_idxs]
        valid_images = np.array(images)[valid_idxs]
        valid_labels = np.array(labels)[valid_idxs]
        test_images = np.array(images)[test_idxs]
        test_labels = np.array(labels)[test_idxs]

        self.train = DataSet(train_images, train_labels, batch_size)
        self.valid = DataSet(valid_images, valid_labels, batch_size)
        self.test = DataSet(test_images, test_labels, batch_size)


def load_face_image(batch_size=128, split=[0.7, 0.0, 0.3], keep_in_memory=True, preload=False):
    rootdir = '../data/FaceImage'
    tarpath = '%s.tar.gz' % rootdir

    if not os.path.exists('../data'):
        os.makedirs('../data')

    if not os.path.exists(rootdir):
        if not os.path.exists(tarpath):
            urllib.urlretrieve('https://s3.amazonaws.com/michalkoziarski/FaceImage.tar.gz', tarpath)

        with tarfile.open(tarpath) as tar:
            tar.extractall('../data')

    genders = ['m', 'f']
    ages = ['(0, 2)', '(4, 6)', '(8, 13)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    dictionary = []

    for g in genders:
        for a in ages:
            dictionary.append('%s_%s' % (g, a))

    dfs = []

    for i in range(5):
        path = os.path.join(rootdir, 'fold_%d_data.txt' % i)
        dfs.append(pd.read_csv(path, sep='\t'))

    df = pd.concat(dfs, ignore_index=True)
    df['path'] = df['user_id'] + '/landmark_aligned_face.' + df['face_id'].astype(str) + '.' + df['original_image']
    df['path'] = df['path'].apply(lambda x: os.path.join(rootdir, 'aligned', x))
    df['age'] = df['age'].map(lambda x: x if x in ages else None)
    df['gender'] = df['gender'].map(lambda x: x if x in genders else None)
    df = df[['path', 'age', 'gender']].dropna()
    df['label'] = df['gender'].astype(str) + '_' + df['age'].astype(str)

    images = []
    labels = []

    for _, row in df.iterrows():
        path, _, _, label = row
        one_hot = np.zeros(len(dictionary))
        one_hot[dictionary.index(label)] = 1
        images.append(Image(path, keep_in_memory=keep_in_memory, preload=preload))
        labels.append(one_hot)

    return DataSets(images, labels, batch_size, split)
