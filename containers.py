import numpy as np
import matplotlib.pyplot as plt

from scipy import misc


class Image:
    def __init__(self, image=None, path=None, shape=None, keep_in_memory=True, preload=False, normalize=True,
                 noise=None, grayscale=False, patch_size=None, sample_size=None, coordinates=None,
                 noise_before_resize=True):
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
        self.patch_size = patch_size
        self.sample_size = sample_size
        self.coordinates = coordinates
        self.noise_before_resize = noise_before_resize
        self.image = None

        if preload or image is not None:
            self.load_and_process(image)

    def get(self):
        if self.image is not None:
            return self.image
        else:
            return self.load_and_process()

    def patch(self, size=None, coordinates=None, return_coordinates=False):
        image = Image(image=self.image, path=self.path, shape=self.shape, keep_in_memory=True, normalize=self.normalize,
                      noise=self.noise, grayscale=self.grayscale, patch_size=size, sample_size=self.sample_size,
                      coordinates=coordinates, noise_before_resize=self.noise_before_resize)
        patch = image.get()

        if return_coordinates:
            return patch, image.coordinates
        else:
            return patch

    def sample(self, size=None, coordinates=None, return_coordinates=False):
        image = Image(image=self.image, path=self.path, shape=self.shape, keep_in_memory=True, normalize=self.normalize,
                      noise=self.noise, grayscale=self.grayscale, patch_size=self.patch_size, sample_size=size,
                      coordinates=coordinates, noise_before_resize=self.noise_before_resize)
        sample = image.get()

        if return_coordinates:
            return sample, image.coordinates
        else:
            return sample

    def load_and_process(self, image=None):
        if image is None:
            image = misc.imread(self.path, mode='RGB')
        else:
            image = np.copy(image)

        if self.noise is not None and self.noise_before_resize:
            if self.normalize and image.dtype == np.dtype('uint8'):
                image = image / 255.

            self.noise.set_scale(self.scale)

            image = self.noise.apply(image)

        if self.shape is not None:
            image = self._resize(image, self.shape)

        if self.patch_size is not None:
            x = image.shape[0] * self.patch_size / np.min(image.shape[0:2])
            y = image.shape[1] * self.patch_size / np.min(image.shape[0:2])

            image = self._resize(image, (x, y))

            if self.coordinates is not None:
                x, y = self.coordinates
            else:
                x = np.random.randint(image.shape[0] - self.patch_size + 1)
                y = np.random.randint(image.shape[1] - self.patch_size + 1)

                self.coordinates = (x, y)

            image = image[x:(x + self.patch_size), y:(y + self.patch_size)]

        if self.sample_size is not None:
            if image.shape[0] < self.sample_size or image.shape[1] < self.sample_size:
                x = image.shape[0] * self.sample_size / np.min(image.shape[0:2])
                y = image.shape[1] * self.sample_size / np.min(image.shape[0:2])

                image = self._resize(image, (x, y))

            if self.coordinates is not None:
                x, y = self.coordinates
            else:
                x = np.random.randint(image.shape[0] - self.sample_size + 1)
                y = np.random.randint(image.shape[1] - self.sample_size + 1)

                self.coordinates = (x, y)

            image = image[x:(x + self.sample_size), y:(y + self.sample_size)]

        if self.normalize and image.dtype == np.dtype('uint8'):
            image = image / 255.

        if self.grayscale and len(np.shape(image)) == 3 and np.shape(image)[2] >= 3:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

            image = 0.2989 * r + 0.5870 * g + 0.1140 * b

        if self.noise is not None and not self.noise_before_resize:
            self.noise.set_scale(self.scale)

            image = self.noise.apply(image)

        if self.keep_in_memory:
            self.image = image

        return image

    def noisy(self, noise, noise_before_resize=True):
        return Image(image=self.image, path=self.path, shape=self.shape, keep_in_memory=True, normalize=self.normalize,
                     noise=noise, grayscale=self.grayscale, patch_size=self.patch_size, sample_size=self.sample_size,
                     coordinates=self.coordinates, noise_before_resize=noise_before_resize)

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
    def __init__(self, images, targets=None, batch_size=50, cutoff=True, offset=None, shuffle=True):
        assert targets is None or len(images) == len(targets)

        self.images = np.array(images)
        self.targets = np.array(targets) if targets else None
        self.batch_size = batch_size
        self.offset = offset
        self.length = len(images)
        self.batches_completed = 0
        self.epochs_completed = 0
        self.current_index = 0

        if shuffle:
            self.shuffle()

        if cutoff:
            self.length -= self.length % self.batch_size
            self.images = self.images[:self.length]

            if self.targets is not None:
                self.targets = self.targets[:self.length]

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

        if self.targets is not None:
            self.targets = self.targets[perm]

    def _create_batch(self, size):
        raise NotImplementedError


class LabeledDataSet(DataSet):
    def __init__(self, images, targets, noise=None, patch=None, batch_size=50, cutoff=True, offset=None,
                 noise_before_resize=True, shuffle=True, network=None):
        self.noise = noise
        self.patch = patch
        self.noise_before_resize = noise_before_resize
        self.network = network

        DataSet.__init__(self, images, targets, batch_size=batch_size, cutoff=cutoff, offset=offset, shuffle=shuffle)

    def _create_batch(self, size):
        if self.noise is not None:
            images = []

            for image in self.images[self.current_index:(self.current_index + size)]:
                if self.network is None:
                    images.append(image.noisy(self.noise, self.noise_before_resize).patch(self.patch))
                else:
                    noisy = image.noisy(self.noise, self.noise_before_resize)

                    if image.normalize:
                        denoised = self.network.output().eval(feed_dict={self.network.x: [noisy.get()]})[0]
                    else:
                        denoisable = noisy.get().astype(np.float) / 255.0
                        denoised = self.network.output().eval(feed_dict={self.network.x: [denoisable]})[0]
                        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

                    images.append(Image(image=denoised, normalize=image.normalize, grayscale=image.grayscale).patch(self.patch))
        elif self.network is not None:
            images = []

            for image in self.images[self.current_index:(self.current_index + size)]:
                if image.normalize:
                    denoised = self.network.output().eval(feed_dict={self.network.x: [image.get()]})[0]
                else:
                    denoisable = image.get().astype(np.float) / 255.0
                    denoised = self.network.output().eval(feed_dict={self.network.x: [denoisable]})[0]
                    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

                images.append(Image(image=denoised, normalize=image.normalize, grayscale=image.grayscale).patch(self.patch))
        else:
            images = [image.patch(self.patch) for image in self.images[self.current_index:(self.current_index + size)]]

        targets = [target.get() for target in self.targets[self.current_index:(self.current_index + size)]]

        images, targets = np.array(images), np.array(targets)

        if self.offset is not None:
            images -= np.array(self.offset, ndmin=1).astype(images.dtype)

        return images, targets


class UnlabeledDataSet(DataSet):
    def __init__(self, images, noise=None, patch=None, sample=None, batch_size=50, cutoff=True, offset=None,
                 noise_before_resize=True, shuffle=True):
        self.noise = noise
        self.patch = patch
        self.sample = sample
        self.noise_before_resize = noise_before_resize

        DataSet.__init__(self, images, batch_size=batch_size, cutoff=cutoff, offset=offset, shuffle=shuffle)

    def _create_batch(self, size):
        images = []
        targets = []

        for i in range(size):
            if self.current_index + i >= self.length:
                break

            target = self.images[self.current_index + i]

            if self.noise:
                image = target.noisy(self.noise, self.noise_before_resize)
            else:
                image = target

            if self.patch:
                image, coordinates = image.patch(self.patch, return_coordinates=True)
                target = target.patch(self.patch, coordinates=coordinates)
            elif self.sample:
                image, coordinates = image.sample(self.sample, return_coordinates=True)
                target = target.sample(self.sample, coordinates=coordinates)
            else:
                image = image.get()
                target = target.get()

            images.append(image)
            targets.append(target)

        images, targets = np.array(images), np.array(targets)

        if self.offset is not None:
            images -= np.array(self.offset, ndmin=1).astype(images.dtype)
            targets -= np.array(self.offset, ndmin=1).astype(targets.dtype)

        return images, targets
