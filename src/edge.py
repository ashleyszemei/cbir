# -*- coding: utf-8 -*-

from __future__ import print_function

# from evaluate import extract_features_from_db
# from DB import Database
# from src.evaluate import extract_features_from_db
# from src.DB import Database

from six.moves import cPickle
import numpy as np
# import scipy.misc
from math import sqrt
import os
import imageio


stride = (1, 1)
n_slice  = 10
h_type   = 'region'
d_type   = 'cosine'

depth    = 5


edge_kernels = np.array([
  [
   # vertical
   [1,-1], 
   [1,-1]
  ],
  [
   # horizontal
   [1,1], 
   [-1,-1]
  ],
  [
   # 45 diagonal
   [sqrt(2),0], 
   [0,-sqrt(2)]
  ],
  [
   # 135 diagnol
   [0,sqrt(2)], 
   [-sqrt(2),0]
  ],
  [
   # non-directional
   [2,-2], 
   [-2,2]
  ]
])

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class Edge(object):

  def histogram(self, input, stride=(2, 2), type=h_type, n_slice=n_slice, normalize=True):
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = imageio.imread(input, pilmode='RGB')
    height, width, channel = img.shape
  
    if type == 'global':
      hist = self._conv(img, stride=stride, kernels=edge_kernels)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, edge_kernels.shape[0]))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._conv(img_r, stride=stride, kernels=edge_kernels)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _conv(self, img, stride, kernels, normalize=True):
    H, W, C = img.shape
    conv_kernels = np.expand_dims(kernels, axis=3)
    conv_kernels = np.tile(conv_kernels, (1, 1, 1, C))
    assert list(conv_kernels.shape) == list(kernels.shape) + [C]  # check kernels size
  
    sh, sw = stride
    kn, kh, kw, kc = conv_kernels.shape
  
    hh = int((H - kh) / sh + 1)
    ww = int((W - kw) / sw + 1)
  
    hist = np.zeros(kn)
  
    for idx, k in enumerate(conv_kernels):
      for h in range(hh):
        hs = int(h*sh)
        he = int(h*sh + kh)
        for w in range(ww):
          ws = w*sw
          we = w*sw + kw
          hist[idx] += np.sum(img[hs:he, ws:we] * k)  # element-wise product
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist
  
  
  def extract_features(self, db, query_image_filepath=None, is_query=False, verbose=True):
    if h_type == 'global':
      db_img_features_cache = "edge-{}-stride{}".format(h_type, stride)
    elif h_type == 'region':
      db_img_features_cache = "edge-{}-stride{}-n_slice{}".format(h_type, stride, n_slice)

    print("Now making Edge samples.")

    if is_query == False:
      # If extracting features from all images in database
      try:
        # Use cached image features
        db_images = cPickle.load(open(os.path.join(cache_dir, db_img_features_cache), "rb", True))
        for db_image in db_images:
          db_image['hist'] /= np.sum(db_image['hist'])  # normalize
        if verbose:
          print("Using cache..., config=%s, distance=%s, depth=%s" % (db_img_features_cache, d_type, depth))
      except:
        if verbose:
          print("Counting histogram..., config=%s, distance=%s, depth=%s" % (db_img_features_cache, d_type, depth))

        db_images = []
        data = db.get_data()
        for d in data.itertuples():
          d_img, d_cls = getattr(d, "img"), getattr(d, "cls")

          # Generate edge histogram for each image in the database
          d_hist = self.histogram(d_img, type=h_type, n_slice=n_slice)
          db_images.append({
            'img': d_img,
            'cls': d_cls,
            'hist': d_hist
          })
        # Cache extracted image features
        cPickle.dump(db_images, open(os.path.join(cache_dir, db_img_features_cache), "wb", True))
        print("Edge samples: ", db_images)
      return db_images
    elif is_query == True:
      # If extracting features from single image
      # Generate edge histogram for the query image
      d_hist = self.histogram(query_image_filepath, type=h_type, n_slice=n_slice)
      query = {
        'img': query_image_filepath,
        'cls': "query",
        'hist': d_hist
      }
      return query
