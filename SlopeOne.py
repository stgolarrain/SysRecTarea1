import operator
import pickle
import math
import numpy as np

from Loader import Loader
from sklearn import cross_validation

FILE_PATH = '../train/ratings.csv'
PREDICT_PATH = '../prediction/prediction.csv'
FILE_PATH2 = '../train/ratings2.csv'
TOPN_PATH = '../prediction/list-top-n.txt'

class SlopeOne(object):
	def __init__(self, loader):
		self.diffs = {}
		self.freqs = {}
		self.data  = loader.data
		self.loader = loader
		self.avg = None
		self.train_idx = None
		self.test_idx = None
		self.item_data = None
		self.k = 10

		self.predict_data = loader.predict

	def similarities(self):
		for u,ratings in self.data.items():
			length = len(ratings.items())
			print str.format('\tProcessing user {0}/{1}', u, len(self.data.items()))
			if  length > 1500:
		 		print 'Too many items'
			else:
				for (item, rating) in ratings.items():
					self.freqs.setdefault(item, {})
					self.diffs.setdefault(item, {})
					for (item2, rating2) in ratings.items():
						if item != item2:
							self.freqs[item].setdefault(item2, 0)
							self.diffs[item].setdefault(item2, 0.0)
							self.freqs[item][item2] += 1
							self.diffs[item][item2] += rating - rating2
		

	def predict(self, u, i, default=True):
		norm = 0
		score = 0
		#print str.format('Predicting User: {0} - Item {1}', u, i)
		
		for item,rating in self.data[u].items():
			if i != item and self.freqs.has_key(i) and self.freqs[i].has_key(item) and self.freqs[i][item] > 0:
				norm += self.freqs[i][item]
				score += (rating + self.diffs[i][item])*self.freqs[i][item]
		if norm > 0:
			return score/norm
		elif default:
			return self.avg[u]
		else:
			return 8

	def top10_idx(self, user):
		predictions = []
		for item in self.data[user].keys():
			score = self.predict(user, item, default=False)
			predictions.append((item, score))
		predictions = sorted(predictions, key=operator.itemgetter(1))
		predictions.reverse()
		return [i for i,s in predictions[0:10]]

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
		count = 0
		for n,item in enumerate(test):
			if n%1000 == 0: print str.format('\tProcessing RMSE item {0}/{1}', n, len(test))
			#print str.format('\tProcessing RMSE item {0}/{1}', n, len(test))
			for user,score in self.item_data[item].items():
				score = self.predict(user, item)
				score = min(score, 10)
				score = max(score, 1)
				RMSE += (score - self.data[user][item])**2
				count += 1
		return math.sqrt(RMSE/count)

	def test_topN(self, test):
		top_users = set()
		precision = 0.
		n = 0
		for n,item in enumerate(test):
			if n%1000 == 0: print str.format('\tProcessing top10 item {0}/{1}', n, len(test))
			for user,rating in self.item_data[item].items():
				if rating >= np.mean(self.data[user].values())+np.std(self.data[user].values()) and not user in top_users:
					top_users.add(user)
					top10 = self.top10_idx(user)
					relevants = self.relevants(user)
					precision += len(np.intersect1d(top10, relevants)) / 10
					n += 1
		return precision / n
		
	def predict_file(self):
		self.loader.load_predict()
		length = len(self.loader.predict.items())
		file = open("results/prediction.csv", "wb")
		for user,items in self.loader.predict.items():
			print str.format('Predicting user {0}/{1}', self.loader.user_idx(user), length)
			for item in items.keys():
				try:
					score = self.predict(self.loader.user_idx(user), self.loader.item_idx(item))
					score = min(score, 10)
					score = max(score, 1)
					file.write(str.format('"{0}";"{1}";{2}\n', user, item, score))
				except Exception:
					score = np.mean(self.data[self.loader.user_idx(user)].values())
					file.write(str.format('"{0}";"{1}";{2}\n', user, item, score))

	def mean(self):
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
		print len(self.avg)

	def top10(self):
		top10 = {}
		length = len(self.loader.predict.items())
		for user,items in self.loader.predict.items():
			print str.format('Predicting top10 user {0}/{1}', self.loader.user_idx(user), length)
			predictions = []
			for item in self.loader.items():
				i = self.loader.item_idx(item)
				u = self.loader.user_idx(user)
				score = self.predict(u, i)
				predictions.append((item, score))
			predictions = sorted(predictions, key=operator.itemgetter(1))
			predictions.reverse()
			top10[user] = predictions[0:10]
		file = open("temp/slope_one/top10.txt", "wb")
		for user,items in top10.items():
			file.write(str.format("{0}\n", user))
			for data in items:
				file.write(str.format('\t"{0}"\n', data[0]))

if __name__ == '__main__':

	loader = Loader(FILE_PATH, PREDICT_PATH)
	loader.load_user_base()
	slope_one = SlopeOne(loader)
	loader2 = Loader(FILE_PATH, PREDICT_PATH)
	loader2.load_item_base()
	slope_one.item_data = loader2.data

	try:
		slope_one.avg = pickle.load(open('temp/slope_one/mean.p', 'rb'))
	except Exception:
		slope_one.mean()
		pickle.dump(slope_one.avg, open('temp/slope_one/mean.p', 'wb'))

	try:
		print '> Loading dev matrix'
		f = open('temp/slope_one/freqs.p', 'rb')
		slope_one.freqs = pickle.load(f)
		f.close()
		f = open('temp/slope_one/diffs.p', 'rb')
		slope_one.diffs = pickle.load(f)
		f.close()
		print '> Loaded Slop One'
	except Exception, e:
		print e
		slope_one.similarities()
		pickle.dump(slope_one.freqs, open('temp/slope_one/freqs.p', 'wb'))
		pickle.dump(slope_one.diffs, open('temp/slope_one/diffs.p', 'wb'))

	# print '> Testing model'
	# kf = cross_validation.KFold(len(slope_one.loader.items), n_folds=5)
	# RMSE = 0.
	# precision = 0.
	# for train_index, test_index in kf:
	# 	slope_one.train_idx = train_index
	# 	slope_one.test_idx = test_index
	# 	RMSE += slope_one.test_error(test_index)
	# 	precision += slope_one.test_topN(test_index)
	# print 'RMSE = ', RMSE/5
	# print 'Precison@10', precision/5

	print '> Predicting File'
	slope_one.predict_file()

	
	


