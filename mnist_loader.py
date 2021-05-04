import io
import numpy as np

class DataLoader:

	def load_labels(self, file_loc, limit):
		with open(file_loc, "rb") as f:
			bytes_read = f.read()
		
		stream = io.BytesIO(bytes_read)
		nums = []
		a = int.from_bytes(stream.read(4), byteorder='big')
		if a == 2049:
			s = int.from_bytes(stream.read(4), byteorder='big')
		
			if limit:
				s = limit
			print(s)
			for i in range(s):
				b = stream.read(1)
				nums.append(self.vectorize(ord(b)))
			return nums
		else:
			print("Error while loading labels: checksum doesn't match")
			return None
	
	def vectorize(self, j):
		e = np.zeros((10,1))
		e[j] = 1
		return e

	def load_images(self, file_loc, limit):
		with open(file_loc, "rb") as f:
			img_bytes = f.read()

		stream = io.BytesIO(img_bytes)
		
		images = []
		a = int.from_bytes(stream.read(4), byteorder='big')
		if a == 2051:
			s = int.from_bytes(stream.read(4), byteorder='big')
			h = int.from_bytes(stream.read(4), byteorder='big')
			w = int.from_bytes(stream.read(4), byteorder='big')
			
			if limit:
				s = limit

			for i in range(s):
				image = []
				for j in range(h*w):
					b = int.from_bytes(stream.read(1), byteorder='big')
					image.append(float(b)/255)
				images.append(np.array([image]).transpose())
			return images
		else:
			print("Error while loading images: checksum doesn't match")
			return None
	
	def load_test_set(self, limit=None):
		return list(zip(self.load_images("data/t10k-images-idx3-ubyte", limit),
						self.load_labels("data/t10k-labels-idx1-ubyte", limit)))
	
	def load_training_set(self, limit=None):
		return list(zip(self.load_images("data/train-images-idx3-ubyte", limit),
						self.load_labels("data/train-labels-idx1-ubyte", limit)))



#with open("data/t10k-images-idx3-ubyte", "rb") as f:
#	img_bytes = f.read()
#
#stream = io.BytesIO(img_bytes)
#
#a = int.from_bytes(stream.read(4), byteorder='big')
#if a == 2051:
#	s = int.from_bytes(stream.read(4), byteorder='big')
#	h = int.from_bytes(stream.read(4), byteorder='big')
#	w = int.from_bytes(stream.read(4), byteorder='big')
#
#	print(s)
#	print(h)
#	print(w)
#	
#	ctr = 0;
#	while ctr < s:
#		img = io.BytesIO()
#		# bitmap header
#		img.write(b"\x42\x4D\x00\x00\x00\x00\x00\x00\x00\x00\x36\x04\x00\x00\x28\x00\x00\x00\x1C\x00\x00\x00\x1C\x00\x00\x00\x01\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\x00\x00\x00\x00\x00\x00\x00")
#		for k in range(256):
#			img.write(bytearray([k,k,k,0]))
#		rows = []
#		for i in range(h):
#			row = stream.read(w)
#			rows.insert(0,row)
#		for i in range(h):
#			img.write(rows[i])
#		
#		out = open(f"img/{ctr:05d}_{nums[ctr]}.bmp", "wb")
#		img.seek(0)
#		out.write(img.read())
#		print(f"Wrote to file img/{ctr:05d}_{nums[ctr]}.bmp")
#		ctr += 1
