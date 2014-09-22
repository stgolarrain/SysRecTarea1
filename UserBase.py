import math
import operator
import pickle
import numpy as np

from scipy.stats import pearsonr
from Loader import Loader
from sklearn import cross_validation


FILE_PATH = '../train/ratings.csv'
FILE_PATH2 = '../train/ratings2.csv'
PREDICT_PATH = '../prediction/prediction.csv'
TOPN_PATH = '../prediction/list-top-n.txt'

class UserBase(object):
	def __init__(self, loader, k):
		self.data = loader.data
		self.loader = loader
		self.pearson = {}
		self.k = k
		self.mean = None
		self.neighbors = {}
		self.avg = None
		self.train_idx = None
		self.test_idx = None

	def similarities(self):
		print '> Loading pearson Corr matrix'
		for u1,user1 in self.data.items():
			print str.format('Computing user {0}/{1}', u1, self.n_user)
			self.pearson.setdefault(u1, {})
			idx1 = user1.keys()
			if len(idx1) > 1:
				for u2 in xrange(u1+1,self.n_user):
					idx2 = self.data[u2].keys()
					if len(idx2) > 1:
						idx = np.intersect1d(idx1, idx2, assume_unique=True)
						if len(idx) > 1:
							scores1 = [self.data[u1][i] for i in idx]
							scores2 = [self.data[u2][i] for i in idx]
							corr = pearsonr(scores1, scores2)[0]
							if not np.isnan(corr):
								corr = np.float16(corr)
								self.pearson.setdefault(u2, {})
								self.pearson[u1][u2] = corr
								self.pearson[u2][u1] = corr
		return self.pearson

	def knn(self):
		print '> Computing User Base KNN'
		for user_id, pearson in self.pearson.items():
			if pearson != {} and (self.test_idx == None or user_id in self.test_idx):
				sorted_u = sorted(pearson.iteritems(), key=operator.itemgetter(1), reverse=True)
				if self.train_idx != None:
					sorted_u = np.intersect1d([i for i,s in sorted_u], self.train_idx, assume_unique=True)
				else:
					sorted_u = [i for i,s in sorted_u]
				self.neighbors[user_id] = sorted_u[0:self.k]

	def generate_mean(self):
		print '> Computing mean user'
		mean = []
		for i,user in self.data.items():
			if i%1000 == 0:
				print str.format('\tProcessing user {0}/{1}', i, self.matrix.shape[0])
			m = np.mean(user.values())
			if np.isnan(mean):
				mean.append(0)
			else:
				mean.append(m)
		self.mean = np.array(mean)
		return self.mean

	def mean_item(self):
		print '> Generating mean Item'
		mean = {}
		n = {}
		for user,rating in self.data.items():
			for i,r in rating.items():
				mean.setdefault(i, 0)
				n.setdefault(i, 0)
				mean[i] += r
				n[i] += 1
		
		self.avg = []
		for i,v in mean.items():
			self.avg.append(v/n[i])

	def predict(self, u, i, default=True):
		mean = self.mean[u]
		if self.neighbors.has_key(u):
			score = 0
			norm  = 0
			for n,neighbor in enumerate(self.neighbors[u]): # data = (userId, pearson)
				if i in self.data[neighbor]:
					corr = self.pearson[u][neighbor]
					norm += corr
					score += corr*(self.data[neighbor][i]-self.mean[neighbor])
			
			if norm > 0:
				return mean + score/norm
			elif default:
				return (mean + self.avg[i])/2
			else:
				return 0
		elif default:
			return (mean + self.avg[i])/2
		else:
			return 0

	def top10_idx(self, user):
		predictions = []
		for item in self.data[user].keys():
			score = self.predict(user, item, default=False)
			predictions.append((item, score))
		predictions = sorted(predictions, key=operator.itemgetter(1))
		predictions.reverse()
		return [i for i,s in predictions[0:10]]

	def top10_list(self, user):
		predictions = []
		for item in list(self.loader.items):
			i = self.loader.item_idx(item)
			u = self.loader.user_idx(user)
			if not self.data[u].has_key(i):
				score = self.predict(u, i, default=False)
				predictions.append((item, score))
		predictions = sorted(predictions, key=operator.itemgetter(1), reverse=True)
		return predictions[0:10]

	def relevants(self, user):
		rel = []
		mean = np.mean(self.data[user].values())
		std = np.std(self.data[user].values())
		for i,score in self.data[user].items():
			if score <= mean+std:
				rel.append(i)
		return rel

	def test_error(self, test):
		RMSE = 0.
		n = 0
		for i,u in enumerate(test):
			if i%1000 == 0: print str.format('\tProcessing user {0}/{1}', i, len(test))
			for item, score in self.data[u].items():
				score = self.predict(u, item)
				if score < 1 or score > 10:
					score = self.mean[u]
				RMSE += (score - self.data[u][item])**2
				n += 1
		return math.sqrt(RMSE/n)
			

	def test_topN(self, test):
		precision = 0.
		n = 0
		for i,u in enumerate(test):
			if i%1000 == 0: print str.format('\tProcessing user {0}/{1}', i, len(test))
			top10 = self.top10_idx(u)
			relevants = self.relevants(u)
			precision += len(np.intersect1d(top10, relevants)) / 10
			n += 1
		return precision / n

	def top10(self):
		users = []
		with open(TOPN_PATH, 'r') as f:
			for line in f:
				u = line.replace('"', '')
				u = u.replace('\n', '')
				users.append(u)

		file = open("results/list-top-n.txt", "wb")
		for user in users:
			predictions = self.top10_list(user)
			file.write(str.format('"{0}"\n', user))
			for item,score in predictions:
				file.write(str.format('\t"{0}"\n', item))
		file.close()

		# top10 = {}
		# self.loa
		# length = len(self.loader.predict.items())
		# for user,items in self.loader.predict.items():
		# 	print str.format('Predicting user {0}/{1}', self.loader.user_idx(user), length)
		# 	predictions = []
		# 	for item in list(self.loader.items):
		# 		i = self.loader.item_idx(item)
		# 		u = self.loader.user_idx(user)
		# 		score = self.predict(u, i)
		# 		predictions.append((item, score))
		# 	predictions = sorted(predictions, key=operator.itemgetter(1))
		# 	predictions.reverse()
		# 	top10[user] = predictions[0:10]
		# file = open("results/list-top-n.txt", "wb")
		# for user,items in top10.items():
		# 	file.write(str.format('"{0}"\n', user))
		# 	for data in items:
		# 		file.write(str.format('\t"{0}"\n', data[0]))


def main():
	loader = Loader(FILE_PATH, PREDICT_PATH)
	loader.load_user_base()
	user_base = UserBase(loader, 5)

	try:
		user_base.pearson = pickle.load(open('temp/user_base/pearson.matrix', 'rb'))
	except Exception:
		user_base.similarities()
		pickle.dump(user_base.pearson, open('temp/user_base/pearson.matrix', 'wb'), protocol=-1)
		print '> PEARSON SAVE'

	try:
		user_base.mean = pickle.load(open('temp/user_base/user_mean.p', 'rb'))
	except Exception:
		user_base.generate_mean()
		pickle.dump(user_base.mean, open('temp/user_base/user_mean.p', 'wb'))

	print '> Testing model'
	# k_values = [5, 10, 20, 30]
	# results = {}
	# results.setdefault('RMSE', [])
	# results.setdefault('Precision', [])
	# results['K'] = k_values

	# user_base.mean_item()
	# kf = cross_validation.KFold(len(user_base.loader.users), n_folds=5)
	# RMSE = 0.
	# precision = 0.

	# for k in k_values:
	# 	user_base.k = k
	# 	for train_index, test_index in kf:
	# 		user_base.test_idx = test_index
	# 		user_base.train_idx = train_index
	# 		user_base.knn()
	# 		RMSE += user_base.test_error(test_index)
	# 		precision += user_base.test_topN(test_index)
	# 	print str.format('RMSE [k={0}] = {1}', k, RMSE/5)
	# 	print str.format('PRECISION@10 [k={0}] = {1}', k, precision/5)
	# 	results['RMSE'].append(RMSE / 5)
	# 	results['Precision'].append(precision / 5)
		
	# print results

	print '> Make predictions'
	user_base.mean_item()
	user_base.knn()
	user_base.top10()





if __name__ == '__main__':
	main()