import numpy as np

a0 = np.zeros((2,1))
a1 = np.zeros((2,1))
a2 = 0

W1 = np.array([[-0.5, 0.5],[0.5, -0.5]])
W2 = np.array([1, 1])

b1 = np.array([[0.0], [0.0]])
b2 = 0.5

def cost(a,e):
	return (a-e)^2

def relu(a):
	return np.maximum(a, 0)

def feval(a0):
	return relu(np.matmul(W2, relu(np.matmul(W1,a0) + b1)) + b2)

def run():
	print(W1)
	print(W2)
	print(b1)
	print(b2)

	print(feval(np.array([[0], [0]])))
	print(feval(np.array([[0], [1]])))
	print(feval(np.array([[1], [0]])))
	print(feval(np.array([[1], [1]])))

if __name__ == "__main__":
	run()
