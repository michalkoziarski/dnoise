import numpy as np
import matplotlib.pyplot as plt

from scipy import misc


class Image:
    def __init__(self, image=None, path=None, shape=None, keep_in_memory=True, preload=False, normalize=True,
                 noise=None, grayscale=False):
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
        self.scale = (0.0, 1.0) if normalize else (0, 255)
        self.grayscale = grayscale
        self.image = None

        if preload or image is not None:
            self.load_and_process(image)

    def get(self):
        if self.image is not None:
            return self.image
        else:
            return self.load_and_process()

    def patch(self, size, coordinates=None):
        image = self.get()

        if image.shape[0] < size or image.shape[1] < size:
            x = image.shape[0] * size / np.min(image.shape[0:2])
            y = image.shape[1] * size / np.min(image.shape[0:2])

            image = self._resize(image, (x, y))

        if coordinates is not None:
            x, y = coordinates
        else:
            x = np.random.randint(image.shape[0] - size + 1)
            y = np.random.randint(image.shape[1] - size + 1)

        return image[x:(x + size), y:(y + size)], (x, y)

    def load_and_process(self, image=None):
        if image is None:
            image = misc.imread(self.path, mode='RGB')
        else:
            image = np.copy(image)

        if self.shape is not None:
            image = self._resize(image, self.shape)

        if self.normalize and image.dtype == np.dtype('uint8'):
            image = image / 255.

        if self.grayscale and len(np.shape(image)) == 3 and np.shape(image)[2] >= 3:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

            image = 0.2989 * r + 0.5870 * g + 0.1140 * b

        if self.noise is not None:
            self.noise.set_scale(self.scale)

            image = self.noise.apply(image)

        if self.keep_in_memory:
            self.image = image

        return image

    def noisy(self, noise):
        return Image(image=self.image, path=self.path, shape=self.shape, keep_in_memory=True, normalize=self.normalize,
                     noise=noise, grayscale=self.grayscale)

    def display(self, path=None, size=None):
        image = self.get()

        if len(image.shape) == 3 and image.shape[2] == 1:
            image = np.squeeze(image, axis=(2,))

        color_map = plt.cm.Greys_r if len(image.shape) == 2 else None

        if size is not None:
            image = misc.imresize(image, size)

        if path is None:
            plt.imshow(image, cmap=color_map)
            plt.axis('off')
            plt.show()
        else:
            plt.imsave(path, image, cmap=color_map)

    def _resize(self, image, shape):
        image = misc.imresize(image, shape)

        if self.normalize and image.dtype == np.dtype('uint8'):
            image = image / 255.

        return image


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


class DataSet:
    def __init__(self, images, targets=None, batch_size=50):
        assert targets is None or len(images) == len(targets)

        self.images = np.array(images)
        self.targets = np.array(targets) if targets else None
        self.batch_size = batch_size
        self.length = len(images)
        self.batches_completed = 0
        self.epochs_completed = 0
        self.current_index = 0
        self.shuffle()

    def batch(self, size=None):
        if size is None:
            size = self.batch_size

        images, targets = self._create_batch(size)

        self.batches_completed += 1
        self.current_index += size

        if self.current_index >= self.length:
            self.current_index = 0
            self.epochs_completed += 1

            self.shuffle()

        return images, targets

    def shuffle(self):
        perm = np.random.permutation(self.length)

        self.images = self.images[perm]

        if self.targets:
            self.targets = self.targets[perm]

    def _create_batch(self, size):
        raise NotImplementedError


class LabeledDataSet(DataSet):
    def __init__(self, images, targets, batch_size=50):
        DataSet.__init__(self, images, targets, batch_size)

    def _create_batch(self, size):
        images = [image.get() for image in self.images[self.current_index:(self.current_index + size)]]
        targets = [target.get() for target in self.targets[self.current_index:(self.current_index + size)]]

        return np.array(images), np.array(targets)


class UnlabeledDataSet(DataSet):
    def __init__(self, images, noise=None, patch=None, batch_size=50):
        self.noise = noise
        self.patch = patch

        DataSet.__init__(self, images, batch_size=batch_size)

    def _create_batch(self, size):
        images = []
        targets = []

        for i in range(size):
            if self.current_index + i >= self.length:
                break

            target = self.images[self.current_index + i]

            if self.noise:
                image = target.noisy(self.noise)
            else:
                image = target

            if self.patch:
                image, coordinates = image.patch(self.patch)
                target, _ = target.patch(self.patch, coordinates=coordinates)
            else:
                image = image.get()
                target = target.get()

            images.append(image)
            targets.append(target)

        return np.array(images), np.array(targets)


class KernelEstimationDataSet(DataSet):
    def __init__(self, images, noise, patch=None, kernel_size=None, batch_size=50):
        self.noise = noise
        self.patch = patch
        self.kernel_size = kernel_size

        DataSet.__init__(self, images, batch_size=batch_size)

    def _create_batch(self, size):
        images = []
        targets = []

        for i in range(size):
            if self.current_index + i >= self.length:
                break

            image = self.images[self.current_index + i].noisy(self.noise)

            if self.patch:
                tensor, _ = image.patch(self.patch)
            else:
                tensor = image.get()

            kernel = image.noise.kernel

            if self.kernel_size is not None:
                assert self.kernel_size >= kernel.shape[0] == kernel.shape[1]

                padded_kernel = np.zeros((self.kernel_size, self.kernel_size))
                start = (self.kernel_size - kernel.shape[0]) / 2
                end = (self.kernel_size + kernel.shape[0]) / 2
                padded_kernel[start:end, start:end] = kernel
                kernel = np.expand_dims(padded_kernel, 2)

            images.append(tensor)
            targets.append(kernel)

        return np.array(images), np.array(targets)
