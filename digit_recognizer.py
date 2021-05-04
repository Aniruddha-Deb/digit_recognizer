from mnist_loader import DataLoader
from network import Network

if __name__ == "__main__":
	loader = DataLoader()
	training_set = loader.load_training_set(5000)
	test_set = loader.load_test_set(1000)

	net = Network([784,100,10])
	net.train(training_set, 30, 10, 3.0, test_data=test_set)
