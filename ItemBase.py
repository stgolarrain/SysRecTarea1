import numpy as np
import pickle
import operator
import math

from Loader import Loader
from scipy.spatial.distance import cosine
from sklearn import cross_validation


FILE_PATH = '../train/ratings.csv'
PREDICT_PATH = '../prediction/prediction.csv'
FILE_PATH2 = '../train/ratings2.csv'
TOPN_PATH = '../prediction/list-top-n.txt'

class ItemBase(object):
	def __init__(self, loader, k):
		self.loader = loader
		self.data = loader.data
		self.ad_data = loader.data
		self.cosine = {}
		self.mean = None
		self.neighbors = {}
		self.k = k
		self.train_idx = None
		self.test_idx = None
		self.user_data = None

	def similarities(self):
		print '> Computing Adjust Cosine Similarity'
		try:
			self.ad_data = pickle.load(open('temp/item_base/ad_items.data', 'rb'))
		except Exception:
			for i,item in self.ad_data.items():
				if i%5000 == 0:
					print str.format('> Adjusting item {0}/{1}', i, len(self.ad_data.items()))
				for u,rating in item.items():
					self.ad_data[i][u] = rating - self.mean[u]
			pickle.dump(self.ad_data, open('temp/item_base/ad_items.data', 'wb'))

		length = len(self.ad_data.items())
		for i1,item1 in self.ad_data.items():
			print str.format('Computing item {0}/{1}', i1, length)
			self.ad_cosine.setdefault(i1, {})
			idx1 = item1.keys()
			if len(idx1) > 1:
				for i2 in xrange(i1+1,length):
					idx2 = self.ad_data[i2].keys()
					if len(idx2) > 1:
						idx = np.intersect1d(idx1, idx2, assume_unique=True)
						if len(idx) > 1:
							scores1 = [self.ad_data[i1][i] for i in idx]
							scores2 = [self.ad_data[i2][i] for i in idx]
							corr = -cosine(scores1, scores2) + 1
							if not np.isnan(corr):
								corr = np.float16(corr)
								self.cosine.setdefault(i2, {})
								self.cosine[i1][i2] = corr
								self.cosine[i2][i1] = corr
		return self.cosine

	def knn(self):
		print '> Computing User Base KNN'
		#length = len(self.cosine.items())
		for item_id, cosine in self.cosine.items():
			#if item_id%10000 == 0: print str.format('\tProcessing item {0}/{1}', item_id, length)
			if cosine != {} and item_id in self.test_idx:
				sorted_i = sorted(cosine.iteritems(), key=operator.itemgetter(1))
				sorted_i.reverse()
				sorted_i = np.intersect1d([i for i,s in sorted_i], self.train_idx, assume_unique=True)
				self.neighbors[item_id] = sorted_i[0:self.k]
		return self.neighbors

	def mean(self):
		print '> Computing mean user'
		self.loader.load_user_base()
		aux = self.loader.data
		mean = []
		for i,user in aux.items():
			if i%1000 == 0:
				print str.format('\tProcessing user {0}/{1}', i, self.matrix.shape[0])
			m = np.mean(user.values())
			if np.isnan(mean):
				mean.append(0)
			else:
				mean.append(m)
		self.mean = np.array(mean)
		return self.mean

	def predict(self, u, i):
		if self.neighbors.has_key(i):
			score = 0
			norm  = 0
			for neighbor in self.neighbors[i]:
				if neighbor in self.user_data[u]:
					norm += self.cosine[i][neighbor]
					score += self.cosine[i][neighbor]*self.user_data[u][neighbor]
			if norm > 0:
				return score/norm
			else:
				return (self.mean[u] + np.mean(self.data[i].values()))/2
		else:
			return (self.mean[u] + np.mean(self.data[i].values()))/2

	def top10_idx(self, user):
		predictions = []
		for item in self.user_data[user].keys():
			score = self.predict(user, item)
			predictions.append((item, score))
		predictions = sorted(predictions, key=operator.itemgetter(1))
		predictions.reverse()
		return [i for i,s in predictions[0:10]]

	def relevants(self, user):
		rel = []
		mean = np.mean(self.user_data[user].values())
		std = np.std(self.user_data[user].values())
		for i,score in self.user_data[user].items():
			if score <= mean+std:
				rel.append(i)
		return rel

	def test_error(self, test):
		RMSE = 0.
		count = 0
		for n,item in enumerate(test):
			if n%1000 == 0: print str.format('\tProcessing RMSE item {0}/{1}', n, len(test))
			for user,score in self.data[item].items():
				score = self.predict(user, item)
				if score < 1 or score > 10:
					score = self.mean[user]
				RMSE += (score - self.data[item][user])**2
				count += 1
		return math.sqrt(RMSE/count)

	def test_topN(self, test):
		top_users = set()
		precision = 0.
		n = 0
		for n,item in enumerate(test):
			if n%1000 == 0: print str.format('\tProcessing top10 item {0}/{1}', n, len(test))
			for user,rating in self.data[item].items():
				if rating >= np.mean(self.user_data[user].values())+np.std(self.user_data[user].values()) and not user in top_users:
					top_users.add(user)
					top10 = self.top10_idx(user)
					relevants = self.relevants(user)
					precision += len(np.intersect1d(top10, relevants)) / 10
					n += 1
		return precision / n

	def predict_file(self, loader2):
		print '> Predicting file'
		self.loader.load_predict()
		RMSE = 0.
		n = 0
		file = open("temp/item_base/prediction.csv", "wb")
		for user,items in self.loader.predict.items():
			for item in items.keys():
				try:
					score = self.predict(self.loader.user_idx(user), self.loader.item_idx(item))
					print score
					if score < 1:
						score = 1
					elif score > 10:
						score = 10
					RMSE += (score-int(loader2.data[loader2.user_idx(user)][loader2.item_idx(item)]))**2
					n += 1
					file.write(str.format('"{0}";"{1}";{2}\n', user, item, score))
				except Exception:
					score = 5
					#score = self.mean[loader2.user_idx(user)]
					RMSE += (score-int(loader2.data[loader2.user_idx(user)][loader2.item_idx(item)]))**2
					n += 1
					file.write(str.format('"{0}";"{1}";{2}\n', user, item, score))
		print math.sqrt(RMSE/n)


	def top10(self):
		top10 = {}
		length = len(self.loader.predict.items())
		for user,items in self.loader.predict.items():
			print str.format('Predicting user {0}/{1}', self.loader.user_idx(user), length)
			predictions = []
			for item in self.loader.items():
				i = self.loader.item_idx(item)
				u = self.loader.user_idx(user)
				score = self.predict(u, i)
				predictions.append((item, score))
			predictions = sorted(predictions, key=operator.itemgetter(1))
			predictions.reverse()
			top10[user] = predictions[0:10]
		file = open("temp/item_base/top10.txt", "wb")
		for user,items in top10.items():
			file.write(str.format("{0}\n", user))
			for data in items:
				file.write(str.format('\t"{0}"\n', data[0]))

def main():
	loader = Loader(FILE_PATH, PREDICT_PATH)
	loader.load_item_base()

	item_base = ItemBase(loader, 5)
	loader2 = Loader(FILE_PATH, PREDICT_PATH)
	loader2.load_user_base()
	item_base.user_data = loader2.data
	try:
		item_base.mean = pickle.load(open('temp/item_base/user_mean.p', 'rb'))
	except Exception:
		item_base.mean()
		pickle.dump(item_base.mean, open('temp/item_base/user_mean.p', 'wb'))

	try:
		item_base.cosine = pickle.load(open('temp/item_base/ad_cosine.matrix', 'rb'))
	except Exception:
		item_base.similarities()
		pickle.dump(item_base.cosine, open('temp/item_base/ad_cosine.matrix', 'wb'))

	
	print '> Testing model'
	k_values = [5, 10, 20, 30]
	results = {}
	results.setdefault('RMSE', [])
	results.setdefault('Precision', [])
	results['K'] = k_values

	kf = cross_validation.KFold(len(item_base.loader.items), n_folds=5)
	for k in k_values:
		item_base.k = k
		RMSE = 0.
		precision = 0.
		for train_index, test_index in kf:
			item_base.train_idx = train_index
			item_base.test_idx = test_index
			item_base.knn()
			RMSE += item_base.test_error(test_index)
			precision += item_base.test_topN(test_index)
		print str.format('RMSE [k={0}] = {1}', k, RMSE/5)
		print str.format('PRECISION@10 [k={0}] = {1}', k, precision/5)
		results['RMSE'].append(RMSE / 5)
		results['Precision'].append(precision / 5)
	print results
	
	


	# loader2 = Loader(FILE_PATH2, PREDICT_PATH)
	# loader2.load_user_base()
	# item_base.predict_file(loader2)





if __name__ == '__main__':
	main()