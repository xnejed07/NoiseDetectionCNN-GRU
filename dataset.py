import pandas as pd
import numpy as np
import copy
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio
import tqdm

class Dataset:
    def __init__(self,path):
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'
        self.df = pd.read_csv(self.path + 'segments.csv')
        self.NFFF = 200

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        sid = self.df.iloc[item]['segment_id']
        target = self.df.iloc[item]['category_id']
        data = sio.loadmat(self.path+'{}'.format(sid))['data']
        _,_, data = signal.spectrogram(data[0,:],fs=5000,nperseg=256,noverlap=128,nfft=1024)

        data = data[:self.NFFF,:]
        data = stats.zscore(data,axis=1)
        data = np.expand_dims(data,axis=0)
        return data,target

    def split_reviewer(self,reviewer_id):
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        idx = self.df['reviewer_id']!=reviewer_id

        train.df = train.df[idx].reset_index(drop=True)
        valid.df = valid.df[np.logical_not(idx)].reset_index(drop=True)
        return train,valid

    def split_random(self,N_valid):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        train.df = train.df.iloc[N_valid:].reset_index(drop=True)
        valid.df = valid.df.iloc[:N_valid].reset_index(drop=True)
        return train,valid

    def integrity_check(self):
        # iterate through dataset and check if all the files might be correctly loaded
        try:
            for i in tqdm.tqdm(range(len(self))):
                x = self.__getitem__(i)
        except Exception as exc:
            raise exc

    def remove_powerline_noise_class(self):
        self.df = self.df[self.df['category_id']!=0]
        self.df['category_id'] = self.df['category_id'] - 1
        self.df = self.df.reset_index(drop=True)
        return self



if __name__ == "__main__":
    dataset_fnusa = Dataset('./DATASET_FNUSA/').integrity_check()
    dataset_mayo = Dataset('./DATASET_MAYO/').integrity_check()
