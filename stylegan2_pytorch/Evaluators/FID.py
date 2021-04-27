from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
import numpy as np
import cv2
from scipy.linalg import sqrtm
import math
import pickle

def update_mean_cov(mean, cov, N, batch):
	batch_N = batch.shape[0]

	x = batch
	N += batch_N
	x_norm_old = batch-mean
	mean = mean + x_norm_old.sum(axis=0)/N
	x_norm_new = batch-mean
	cov = ((N-batch_N)/N)*cov + x_norm_old.T.dot(x_norm_new)/N

	return (mean, cov, N)

def frechet_distance(mean1, cov1, mean2, cov2):
	ssdiff = np.sum((mean1 - mean2)**2.0)
	covmean = sqrtm(cov1.dot(cov2))
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	fid = ssdiff + np.trace(cov1 + cov2 - 2.0 * covmean)
	return fid

class FrechetInceptionDistance(object):
	# sizes <= 64: resize_to: 75
	# size = 128: resize to: 139
	# good sizes to resize to: 299 - 32 * n
	
	def __init__(self, mean, cov, image_range=(-1,1), resize_to=75):

		self._inception_v3 = None
		self.image_range = image_range
		self._channels_axis = \
			-1 if K.image_data_format()=="channels_last" else -3

		if self._inception_v3 is None:
			self._setup_inception_network()

		self.real_mean = mean
		self.real_cov = cov

		self.resize_to = resize_to

	def _setup_inception_network(self):
		self._inception_v3 = InceptionV3(
			include_top=False, pooling='avg')
		self._pool_size = self._inception_v3.output_shape[-1]

	def _preprocess(self, images):
		if self.image_range != (-1,1):
			images = images - self.image_range[0]
			images /= (self.image_range[1]-self.image_range[0])/2.0
			images -= 1.0
		if images.shape[self._channels_axis] == 1:
			images = np.concatenate([images]*3, axis=self._channels_axis)
		
		#resize
		resized_images = np.zeros((images.shape[0], self.resize_to, self.resize_to, 3))
		for i in range(images.shape[0]):
			img = images[i]
			img = cv2.resize(img, dsize=(self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR)
			resized_images[i] = img

		return resized_images

	def stats(self, inputs, batch_size=64):
		mean = np.zeros(self._pool_size)
		cov = np.zeros((self._pool_size,self._pool_size))
		N = 0

		for i in range(int(math.ceil(inputs.shape[0]/batch_size))):
			batch = inputs[i*batch_size:min(inputs.shape[0], (i+1)*batch_size)]

			batch = self._preprocess(batch)
			print('fid batch',i)
			pool = self._inception_v3.predict(batch, batch_size=batch_size)
			(mean, cov, N) = update_mean_cov(mean, cov, N, pool)

		return (mean, cov)

	def __call__(self,
			fake_images,
			batch_size=64
		):

		(gen_mean, gen_cov) = self.stats(fake_images, batch_size=batch_size)

		return frechet_distance(self.real_mean, self.real_cov, gen_mean, gen_cov)

class FID:
	def __init__(self, samples=5000, resize_to=75, batch_size=32, real_mean_cov_file='files/cifar10_stats.pickle'):
		real_stats = pickle.load(open(real_mean_cov_file, 'rb'))
		self.fd = FrechetInceptionDistance(real_stats['mean'], real_stats['cov'], (-1,1), resize_to)

		self.samples = samples
		self.batch_size = batch_size

		self.name = 'FID'

	def calculate(self, model, dataset):
		noise = np.random.normal(0, 1, (self.samples, model.latent_dim))
		images = model.generator.predict(noise)
		
		gan_fid = self.fd(images, batch_size=self.batch_size)

		return float(gan_fid)

if __name__ == '__main__':
	import sys
	sys.path.append('../')
	sys.path.append('./')
	import Datasets
	import pickle
	
	dataset = Datasets.CelebA(size=64)
	fd = FrechetInceptionDistance(None, None, (-1,1), 75)
	mean, cov = fd.stats(dataset.X, batch_size=32)
	print(mean,cov)
	
	save = {'mean': mean, 'cov': cov}

	with open('celeba_64_stats.pickle', 'wb') as handle:
		pickle.dump(save, handle)
