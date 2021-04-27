import numpy as np
from keras.models import load_model
import os

class MNIST_IS:
	def __init__(self, samples=100, classifier_file='files/mnist_classifier'):
		self.eps = 1e-16
		self.samples = samples
		#script_dir = os.path.dirname(__file__)
		#path = os.path.join(script_dir, 'mnist_classifier')
		self.classifier_model = load_model(classifier_file)
		self.name = 'InceptionScore'

	def calculate(self, model, dataset):
		# predict class probabilities for images
		noise = np.random.normal(0, 1, (self.samples, model.latent_dim))
		images = model.generator.predict(noise)
		yhat = self.classifier_model.predict(images)

		p_yx = yhat
		# calculate p(y)
		p_y = np.expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (np.log(p_yx + self.eps) - np.log(p_y + self.eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = np.mean(sum_kl_d)
		# undo the log
		is_score = np.exp(avg_kl_d)

		return float(is_score)