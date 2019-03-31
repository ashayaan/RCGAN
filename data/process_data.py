import pandas as pd
import numpy as np
import os



def getFilesName(path):
	return os.listdir(path)

def readFiles(files):
	data_frames = {}
	for name in files:
		if name.endswith('.csv'):
			df = pd.read_csv(path+'/'+name)
			data_frames[name] = df
	return data_frames

def calculateReturns(data_frames):
	for name in data_frames.keys():
		data_frames[name]['Return'] =  np.log(data_frames[name]['Price']) - np.log(data_frames[name]['Open'])
	return data_frames


def mergeReturns(data_frames,files):
	data = {}
	for name in data_frames.keys():
		column_name = name.split('.')[0]
		column_name = column_name.split('_')[0]
	 	data[column_name] = list(data_frames[name]['Return'])


	df = pd.DataFrame.from_dict(data)
	df['Date'] = data_frames['BHEL_Historical_Data.csv']['Date']
	df.to_csv('combined.csv',index=False)	

if __name__ == '__main__':
	path = 'daily_data'
	files = getFilesName(path)

	data_frames = readFiles(files)	

	data_frames = calculateReturns(data_frames)

	mergeReturns(data_frames,files)