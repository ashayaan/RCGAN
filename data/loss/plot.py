import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


def readFile(file):
	return pd.read_csv(file)


if __name__ == '__main__':
	file = 'loss.csv'
	df = readFile(file)
	df.plot(kind='line',x = '0', y='1',color='red')
	plt.show()