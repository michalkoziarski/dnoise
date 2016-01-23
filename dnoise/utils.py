import matplotlib.pyplot as plt

from noise import *
from scipy import misc


class Image:
    def __init__(self, image=None, path=None, shape=None, keep_in_memory=True, preload=False, normalize=True,
                 noise=None, scale=(0, 255)):
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
        self.scale = scale
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
        else:
            image = np.copy(image)

        if self.shape is not None:
            image = misc.imresize(image, self.shape)

        if self.normalize and self.scale[1] != 1.0:
            self.scale = (0, 1.0)

            image = image / 255.

        if self.noise is not None:
            self.noise.set_scale(self.scale)

            image = self.noise.apply(image)

        if self.keep_in_memory:
            self.image = image

        return image

    def noisy(self, noise=GaussianNoise()):
        return Image(image=self.image, path=self.path, shape=self.shape, keep_in_memory=True, normalize=self.normalize,
                     noise=noise, scale=self.scale)

    def display(self, path=None, size=None):
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
    def __init__(self, images, targets, noise=None):
        if noise is not None:
            self._images = [image.noisy(noise) for image in images]
        else:
            self._images = images

        self._targets = targets
        self.noise = noise

    def images(self):
        return np.array([image.get() for image in self._images])

    def targets(self):
        return np.array([target.get() for target in self._targets])

    def noisy(self, noise=GaussianNoise()):
        return Batch(images=self._images, targets=self._targets, noise=noise)

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

        train_idxs = np.random.choice(idxs, train_len, replace=False)
        idxs = [idx for idx in idxs if idx not in train_idxs]
        valid_idxs = np.random.choice(idxs, valid_len, replace=False)
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
