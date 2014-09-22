import csv

FILE_PATH = '../train/ratings.csv'
PREDICT_PATH = '../prediction/prediction.csv'
FILE_PATH2 = '../train/ratings2.csv'
TOPN_PATH = '../prediction/list-top-n.txt'

class Loader:
	def __init__(self, file_path, prediction_path):
		self.file_path = file_path
		self.data = {}
		self.users = set()
		self.items = set()
		self.map_user = {}
		self.map_item = {}
		
		self.predict = {}
		self.prediction_path = prediction_path

	def load_item_base(self):
		print '> Loader: loading data set as dictionary Item-Base'
		with open(self.file_path, 'rb') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=';')
			for (user,item,rating) in csvreader:
				self.map_user.setdefault(user, len(self.users))
				self.map_item.setdefault(item, len(self.items))
				self.users.add(user)
				self.items.add(item)
				self.data.setdefault(self.item_idx(item), {})
				self.data[self.item_idx(item)][self.user_idx(user)] = int(rating)
	
	def load_user_base(self):
		print '> Loader: loading data set as dictionary User-Base'
		with open(self.file_path, 'rb') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=';')
			for (user,item,rating) in csvreader:
				self.map_user.setdefault(user, len(self.users))
				self.map_item.setdefault(item, len(self.items))
				self.users.add(user)
				self.items.add(item)
				self.data.setdefault(self.user_idx(user), {})
				self.data[self.user_idx(user)][self.item_idx(item)] = int(rating)

	def load_predict(self):
		print '> Loader: loading prediction data'
		with open(self.prediction_path, 'rb') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=';')
			for (user,item,rating) in csvreader:
				self.predict.setdefault(user, {})
				self.predict[user][item] = int(rating)

	def histo(self):
		histo = [0 for i in xrange(0,10)]
		print '> Loader: loading data set as dictionary Item-Base'
		with open(self.file_path, 'rb') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=';')
			for (user,item,rating) in csvreader:
				histo[int(rating)-1] += 1
		return histo


	def user_idx(self, user):
		return self.map_user[user]

	def item_idx(self, item):
		return self.map_item[item]

if __name__ == '__main__':
	loader = Loader(FILE_PATH, PREDICT_PATH)
	print loader.histo()
		
		
