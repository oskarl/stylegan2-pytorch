import numpy as np
from keras.models import load_model

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class DiscriminatorAccuracy:
	def __init__(self, samples=100):
		self.samples = samples
		self.name = 'DiscriminatorAccuracy'

	def calculate(self, model, dataset):
		real_images = dataset.batch(self.samples)

		noise = np.random.normal(0, 1, (self.samples, model.latent_dim))
		fake_images = model.generator.predict(noise)
		
		real = model.discriminator.predict(real_images)
		fake = model.discriminator.predict(fake_images)

		if not model.sigmoid:
			real = sigmoid(real)
			fake = sigmoid(fake)

		acc = (np.count_nonzero(real > 0.5) + np.count_nonzero(fake < 0.5)) / (self.samples*2)
		return float(acc)