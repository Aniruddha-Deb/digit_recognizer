import numpy as np

class Network:
	
	# this is all mostly taken from Nielsen's book
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.rand(y,1)*2-1 for y in sizes[1:]]
		self.weights = [np.random.rand(y,x)*2-1 for x,y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = relu(np.dot(w, a)+b)
		return a
	
	def backpropagate(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		# z_i = w_ia_i + b_i
		z_vecs = []
		# feedforward and record activations
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			z_vecs.append(z)
			activation = relu(z)
			#print("RELU(Z)")
			#print(relu(z))
			activations.append(activation)

		# backpropagate
		#print("RELU_PRIME:")
		#print(relu_prime(z_vecs[-1]))
		#print("Z_VECS:")
		#print(z_vecs)
		#print("Activations:")
		#print(activations)
		delta = cost_dv(activations[-1], y)*relu_prime(z_vecs[-1])
		#print("DELTA:")
		#print(delta)
		nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # wow!

		for l in range(2, self.num_layers):
			z = z_vecs[-l]
			# just doing the same thing as before, only iteratively now
			delta = np.dot(self.weights[-l+1].transpose(), delta)*relu_prime(z)
			#print("DELTA:")
			#print(delta)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

		#print("NABLA_B:")
		#print(nabla_b)
		#print("NABLA_W:")
		#print(nabla_w)

		return (nabla_b, nabla_w)

	def train(self, training_data, epochs, batch_size, eta, test_data=None):
		
		if test_data:
			n_test = len(test_data)
		n = len(training_data)

		for j in range(epochs):
			np.random.shuffle(training_data)
			batches = [training_data[k:k+batch_size] for k in range(0,n,batch_size)]
			for batch in batches:
				self.update(batch, eta)

			if test_data:
				print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
			else:
				print(f"Epoch {j} complete")

	def update(self, batch, eta):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x,y in batch:
			delta_nabla_b, delta_nabla_w = self.backpropagate(x,y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

		n = len(batch)
		self.weights = [w-(eta/n)*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/n)*nb for b, nb in zip(self.biases, nabla_b)]

	def evaluate(self, test_data):
		results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in test_data]
		return sum(int(x==y) for (x,y) in results)

def cost_dv(output, y):
	return output-y

def relu(x):
	return 1.0/(1.0 + np.exp(-x))
	# return np.maximum(x,0)

def relu_prime(x):
	return relu(x)*(1-relu(x))
	#return np.greater(x,0).astype(int)

