import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


kpi_ids = [
	'02e99bd4f6cfb33f', '9bd90500bfd11edb', 'da403e4e3f87c9e0',
	'a5bf5d65261d859a', '18fbb1d5a5dc099d', '09513ae3e75778a3',
	'c58bfcbacb2822d1', '1c35dbf57f55f5e4', '046ec29ddf80d62e',
	'07927a9a18fa19ae', '54e8a140f6237526', 'b3b2e6d1a791d63a',
	'8a20c229e9860d0c', '769894baefea4e9e', '76f4550c43334374',
	'e0770391decc44ce', '8c892e5525f3e491', '40e25005ff8992bd',
	'cff6d3c01e6a6bfa', '71595dd7171f4540', '7c189dd36f048a6c',
	'a40b1df87e3f1c87', '8bef9af9a922e0b3', 'affb01ca2b4f0b45',
	'9ee5879409dccef9', '88cf3a776ba00e7c'
]


def vec_kpi(kpi_id):
	vec = np.zeros((len(kpi_ids)))
	vec[kpi_ids.index(kpi_id)] = 1
	return vec


class KPIDataset(Dataset):
	
	def __init__(self, path, seq_length=31, step_width=1, transform=None, evaluate=False):
		assert seq_length % 2 == 1 and seq_length >= 3, "seq_length has to be odd and >= 3."

		self.df = pd.read_csv(path)

		# extract length and mean of KPI IDs
		ids = self.df['KPI ID'].unique()
		self.kpi_lengths = []
		self.kpi_vectors= []

		for _id in ids:
			self.kpi_lengths.append(len(self.df[self.df['KPI ID'] == _id]))
			self.kpi_vectors.append(vec_kpi(_id))

		self.length = sum(self.kpi_lengths)
		self.seq_length = seq_length
		self.step_width = step_width
		self.transform = transform
		self.evaluate = evaluate
		
	def __len__(self):
		return math.ceil(self.length / self.step_width)
	
	def __getitem__(self, index):
		index_df = index * self.step_width

		# find out which KPI ID the index belongs to
		kpi_id = 0
		i = index_df
		for kpi_id, length in enumerate(self.kpi_lengths):
			if i < length:
				break
			i -= length

		# i is index inside correct KPI ID
		start_index = index_df - self.seq_length // 2
		end_index = index_df + self.seq_length // 2 + 1
		kpi_start = index_df - i
		kpi_end = index_df + (self.kpi_lengths[kpi_id] - i)
		
		pad_left_value = np.array([])
		pad_right_value = np.array([])
		pad_left_diff = np.array([])
		pad_right_diff = np.array([])

		if start_index < kpi_start:
			# pad left
			pad_left_value = np.array([self.df.iloc[kpi_start]['value']] * (kpi_start - start_index))
			pad_left_diff = np.array([self.df.iloc[kpi_start]['value_diff']] * (kpi_start - start_index))
			start_index = kpi_start

		if end_index > kpi_end:
			# pad right
			pad_right_value = np.array([self.df.iloc[kpi_end - 1]['value']] * (end_index - kpi_end))
			pad_right_diff = np.array([self.df.iloc[kpi_end - 1]['value_diff']] * (end_index - kpi_end))
			end_index = kpi_end

		rows = self.df.iloc[start_index : end_index]
		x = np.concatenate([
			self.kpi_vectors[kpi_id],
			pad_left_value, 
			rows['value'].values, 
			pad_right_value,
			pad_left_diff,
			rows['value_diff'].values,
			pad_right_diff
		]).astype('float32')

		if self.transform:
			x = self.transform(x)
			
		if self.evaluate:
			return x
		
		y = self.df.iloc[[index_df]]['label'].values[0].astype('int64')
			
		return x, y