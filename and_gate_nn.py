import numpy as np

class Network:
	
	# this is all mostly taken from Nielsen's book
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.rand(y,1) for y in sizes[1:]]
		self.weights = [np.random.rand(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

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
			activations.append(activation)

		# backpropagate
		delta = cost_dv(activations[-1], y)*relu_prime(z_vecs[-1])
		nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # wow!

		for l in range(2, self.num_layers):
			z = z_vecs[-1]
			# just doing the same thing as before, only iteratively now
			delta = np.dot(self.weights[-l+1].transpose(), delta)*relu_prime(z)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

		return (nabla_b, nabla_w)

	def train(self, r):
		print("training")
		training_data = [
			(np.array([[0,0]]).transpose(), 0),
			(np.array([[0,1]]).transpose(), 0),
			(np.array([[1,0]]).transpose(), 0),
			(np.array([[1,1]]).transpose(), 1)
		]

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x,y in training_data:
			delta_nabla_b, delta_nabla_w = self.backpropagate(x,y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

		print(f"{nabla_w}")
		self.weights = [w-(r/4)*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(r/4)*nb for b, nb in zip(self.biases, nabla_b)]
	
	def test(self):
		test_data = [
			np.array([[0,0]]).transpose(),
			np.array([[0,1]]).transpose(),
			np.array([[1,0]]).transpose(),
			np.array([[1,1]]).transpose()
		]

		print("Testing:")
		for data in test_data:
			answer = self.feedforward(data)
			print(f"{data[0,0]} & {data[1,0]} = {answer[0,0]:.2f}")

def cost_dv(output, y):
	return output-y

def relu(x):
	return np.maximum(x,0)

def relu_prime(x):
	return np.greater(x,0).astype(int)

if __name__ == "__main__":
	network = Network((2,2,1))

	for i in range(1,100):
		network.train(0.2)
		network.test()
