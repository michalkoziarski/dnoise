import os
import urllib
import tarfile
import zipfile
import cPickle
import pandas as pd

from utils import *


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


def load_gtsrb(batch_size=128, split=(0.6, 0.2, 0.2), shape=(32, 32), keep_in_memory=True, preload=False,
               train_noise=None, valid_noise=None, test_noise=None):
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

    return DataSets(images, targets, batch_size, split, train_noise=train_noise, valid_noise=valid_noise,
                    test_noise=test_noise)
