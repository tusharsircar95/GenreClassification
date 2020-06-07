import numpy as np
import torch
import glob
import random
import torchvision.transforms as transforms
from dataset_utils import load_song, convert_signal_to_spec, split_signal

class MusicDataset(torch.utils.data.Dataset):
  def __init__(self,song_data,mode,preloaded=False):

    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]

    if preloaded is False:
      max_samples = 660000

      self.song_data = song_data
      self.X_data, self.y_data = [],[]

      for data in self.song_data:
        signal = load_song(data[0])[:max_samples]
        signal_splits = split_signal(signal)
        for split in signal_splits:
          spec = convert_signal_to_spec(split)

          # Shape changes to make it appear as an image
          spec = spec.reshape(spec.shape + (1,))
          #spec = np.concatenate([spec,spec,spec],axis=2)
          spec = spec.transpose(2,0,1)
          
          spec = (spec - np.min(spec))/(np.max(spec)-np.min(spec))
          for i in range(1):
            spec[i] = (spec[i] - self.mean[i]) / self.std[i]

          self.X_data.append(spec)
          self.y_data.append(data[1])

      self.X_data = np.array(self.X_data)
      self.y_data = np.array(self.y_data)
      np.save('./X_{}.npy'.format(mode),self.X_data)
      np.save('./y_{}.npy'.format(mode),self.y_data)
        
    else:
      self.X_data = np.load('./X_{}.npy'.format(mode))
      self.y_data = np.load('./y_{}.npy'.format(mode))
      print('Loaded {} data with shape: '.format(mode),self.X_data.shape)
    
    self.N = len(self.X_data)

  def __len__(self):
    return len(self.X_data)
  
  def __getitem__(self,idx):
      return torch.tensor(np.concatenate([self.X_data[idx],self.X_data[idx],self.X_data[idx]],0).astype(np.float32)), torch.tensor(self.y_data[idx].astype(np.float32))
          




def prepare_data(ROOT='../genres/genres',seed=1):
  
  print('Genres Found: ')
  genres = [ path.split('/')[-2] for path in glob.glob('{}/*/'.format(ROOT)) ]
  print(genres,'\n')

  print('Genre Counts')
  genre_to_idx = {}
  idx_to_genre = {}
  for i,genre in enumerate(genres):
    music_files = glob.glob('{}/{}/*.wav'.format(ROOT,genre))
    genre_to_idx[genre] = i
    idx_to_genre[i] = genre
    print('{} - {}'.format(genre,str(len(music_files))))

  song_data = []
  for genre in genres:
    for path in glob.glob('{}/{}/*.wav'.format(ROOT,genre)):
      song_data.append((path,genre_to_idx[genre]))
  
  random.seed(seed)
  random.shuffle(song_data)
  return song_data


def train_test_split(song_data,split_ratio=0.70):
  n_songs = len(song_data)
  train_split = song_data[:int(split_ratio*n_songs)]
  val_split = song_data[int(split_ratio*n_songs):]
  return train_split, val_split

