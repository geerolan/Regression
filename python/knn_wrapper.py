from run_knn import run_knn
import numpy as np
import matplotlib.pyplot as plt 
import sys

if __name__ == '__main__':
	#import test data
	T = np.load(sys.argv[1])['train_inputs']
	V = np.load(sys.argv[2])['valid_inputs']
	L = np.load(sys.argv[1])['train_targets']
	S = np.load(sys.argv[2])['valid_targets']
	k = [1,3,5,7,9]
	R = list()

	for e in k:
		result = run_knn(e, T, L, V)
		correct = 0.0

		for i in range(len(result)):
			if(result[i] == S[i]):
				correct += 1

		R.append(correct / len(result))

	plt.plot(k, R)
	plt.ylabel('fraction of correct classifications')
	plt.xlabel('number of nearest neighbors')
	plt.show()