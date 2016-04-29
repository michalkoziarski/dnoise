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

    return DataSets.split(images, targets, batch_size, split)


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

    return DataSets.split(images, targets, batch_size, split)


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

    return DataSets.split(images, targets, batch_size, split)


def load_gtsrb(batch_size=128, shape=(32, 32), keep_in_memory=True, preload=False, train_noise=None, test_noise=None):
    root_path = '../data'
    data_path = os.path.join(root_path, 'GTSRB')
    train_img_path = os.path.join(data_path, 'Final_Training', 'Images')
    test_img_path = os.path.join(data_path, 'Final_Test', 'Images')
    annotations_path = os.path.join(root_path, 'GT-final_test.csv')
    train_zip_path = os.path.join(root_path, 'GTSRB_Final_Training_Images.zip')
    test_zip_path = os.path.join(root_path, 'GTSRB_Final_Test_Images.zip')
    annotations_zip_path = os.path.join(root_path, 'GTSRB_Final_Test_GT.zip')
    train_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    test_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip'
    annotations_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(data_path):
        for zip_path, url in [[train_zip_path, train_url], [test_zip_path, test_url], [annotations_zip_path, annotations_url]]:
            if not os.path.exists(zip_path):
                urllib.urlretrieve(url, zip_path)

            with zipfile.ZipFile(zip_path) as z:
                z.extractall(root_path)

    train_images = []
    test_images = []
    train_targets = []
    test_targets = []

    class_dirs = [o for o in os.listdir(train_img_path) if os.path.isdir(os.path.join(train_img_path, o))]

    for class_dir in class_dirs:
        label = int(class_dir)
        class_path = os.path.join(train_img_path, class_dir)
        paths = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.ppm')]

        for path in paths:
            train_images.append(Image(path=path, shape=shape, keep_in_memory=keep_in_memory, preload=preload))
            train_targets.append(Label(label, length=43))

    annotations = pd.read_csv(annotations_path, sep=';')

    for f in os.listdir(test_img_path):
        if not f.endswith('.ppm'):
            continue

        path = os.path.join(test_img_path, f)
        label = annotations[annotations['Filename'] == f]['ClassId'].iloc[0]
        test_images.append(Image(path=path, shape=shape, keep_in_memory=keep_in_memory, preload=preload))
        test_targets.append(Label(label, length=43))

    train_set = DataSet(train_images, train_targets, batch_size, train_noise)
    test_set = DataSet(test_images, test_targets, batch_size, test_noise)

    datasets = DataSets([], [])
    datasets.train = train_set
    datasets.test = test_set
    datasets.valid = None
    datasets.length = train_set.length + test_set.length

    return datasets


def load_stl(batch_size=128, shape=(96, 96), grayscale=True, normalize=True, train_noise=None, test_noise=None,
             n=None):
    root_path = '../data'
    data_path = os.path.join(root_path, 'stl10_binary')
    tar_path = os.path.join(root_path, 'stl10_binary.tar.gz')
    url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(data_path):
        if not os.path.exists(tar_path):
            urllib.urlretrieve(url, tar_path)

        with tarfile.open(tar_path) as tar:
            tar.extractall(root_path)

    def load_images(path, n=None):
        result = []

        with open(os.path.join(data_path, path), 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)

            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 3, 2, 1))

        for image in images:
            result.append(Image(image=image, shape=shape, keep_in_memory=True, grayscale=grayscale,
                                normalize=normalize))

            if n is not None and len(result) >= n:
                return result

        return result

    def load_targets(path, n=None):
        result = []

        with open(os.path.join(data_path, path), 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)

        for label in labels:
            result.append(Label(label - 1, length=10))

            if n is not None and len(result) >= n:
                return result

        return result

    train_images = load_images('train_X.bin', n)
    test_images = load_images('test_X.bin', n)
    train_targets = load_targets('train_y.bin', n)
    test_targets = load_targets('test_y.bin', n)

    train_set = DataSet(train_images, train_targets, batch_size, train_noise)
    test_set = DataSet(test_images, test_targets, batch_size, test_noise)

    datasets = DataSets([], [])
    datasets.train = train_set
    datasets.test = test_set
    datasets.valid = None
    datasets.length = train_set.length + test_set.length

    return datasets


def load_stl_denoised(denoised_path, batch_size=128, shape=(96, 96), grayscale=True, normalize=True,
                      train_noise=None, test_noise=None, n=None):
    datasets = load_stl(batch_size, shape, grayscale, normalize, train_noise, test_noise, n)

    for i in range(datasets.test.length):
        datasets.test._images[i] = Image(path=os.path.join(denoised_path, 'denoised_image_%d.jpg' % (i + 1)),
                                         shape=shape, keep_in_memory=True, grayscale=grayscale, normalize=normalize)

    return datasets


def load_stl_unsupervised(batch_size=128, shape=(96, 96), grayscale=True, normalize=True, train_noise=None,
                          test_noise=None):
    root_path = '../data'
    data_path = os.path.join(root_path, 'stl10_binary')
    tar_path = os.path.join(root_path, 'stl10_binary.tar.gz')
    url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(data_path):
        if not os.path.exists(tar_path):
            urllib.urlretrieve(url, tar_path)

        with tarfile.open(tar_path) as tar:
            tar.extractall(root_path)

    def load_images(path):
        result = []

        with open(os.path.join(data_path, path), 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)

            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 3, 2, 1))

        for image in images:
            result.append(Image(image=image, shape=shape, keep_in_memory=True, grayscale=grayscale,
                                normalize=normalize))

        return result

    train_images = load_images('unlabeled_X.bin')
    test_images = load_images('train_X.bin')
    train_targets = train_images
    test_targets = test_images

    train_set = DataSet(train_images, train_targets, batch_size, train_noise)
    test_set = DataSet(test_images, test_targets, batch_size, test_noise)

    datasets = DataSets([], [])
    datasets.train = train_set
    datasets.test = test_set
    datasets.valid = None
    datasets.length = train_set.length + test_set.length

    return datasets
