# -*- coding: utf-8 -*-

from __future__ import print_function

# from evaluate import *
# from DB import Database
from src.evaluate import *
from src.DB import Database

from skimage.filters import gabor_kernel
from skimage import color
from scipy import ndimage as ndi

import multiprocessing

from six.moves import cPickle
import numpy as np
import scipy.misc
import os


theta     = 4
frequency = (0.1, 0.5, 0.8)
sigma     = (1, 3, 5)
bandwidth = (0.3, 0.7, 1)

n_slice  = 2
h_type   = 'global'
d_type   = 'cosine'

depth    = 1


def make_gabor_kernel(theta, frequency, sigma, bandwidth):
  kernels = []
  for t in range(theta):
    t = t / float(theta) * np.pi
    for f in frequency:
      if sigma:
        for s in sigma:
          kernel = gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s)
          kernels.append(kernel)
      if bandwidth:
        for b in bandwidth:
          kernel = gabor_kernel(f, theta=t, bandwidth=b)
          kernels.append(kernel)
  return kernels


gabor_kernels = make_gabor_kernel(theta, frequency, sigma, bandwidth)
if sigma and not bandwidth:
  assert len(gabor_kernels) == theta * len(frequency) * len(sigma), "kernel nums error in make_gabor_kernel()"
elif not sigma and bandwidth:
  assert len(gabor_kernels) == theta * len(frequency) * len(bandwidth), "kernel nums error in make_gabor_kernel()"
elif sigma and bandwidth:
  assert len(gabor_kernels) == theta * len(frequency) * (len(sigma) + len(bandwidth)), "kernel nums error in make_gabor_kernel()"
elif not sigma and not bandwidth:
  assert len(gabor_kernels) == theta * len(frequency), "kernel nums error in make_gabor_kernel()"

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class Gabor(object):  
  
  def gabor_histogram(self, input, type=h_type, n_slice=n_slice, normalize=True):
    # Generate Gabor histogram
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = scipy.misc.imread(input, mode='RGB')
    height, width, channel = img.shape
  
    if type == 'global':
      hist = self._gabor(img, kernels=gabor_kernels)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, len(gabor_kernels)))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._gabor(img_r, kernels=gabor_kernels)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _feats(self, image, kernel):
    '''
      arguments
        image : ndarray of the image
        kernel: a gabor kernel
      return
        a ndarray whose shape is (2, )
    '''
    feats = np.zeros(2, dtype=np.double)
    filtered = ndi.convolve(image, np.real(kernel), mode='wrap')
    feats[0] = filtered.mean()
    feats[1] = filtered.var()
    return feats
  
  
  def _power(self, image, kernel):
    '''
      arguments
        image : ndarray of the image
        kernel: a gabor kernel
      return
        a ndarray whose shape is (2, )
    '''
    image = (image - image.mean()) / image.std()  # Normalize images for better comparison.
    f_img = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    feats = np.zeros(2, dtype=np.double)
    feats[0] = f_img.mean()
    feats[1] = f_img.var()
    return feats
  
  
  def _gabor(self, image, kernels=make_gabor_kernel(theta, frequency, sigma, bandwidth), normalize=True):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
  
    img = color.rgb2gray(image)
  
    results = []
    feat_fn = self._power
    for kernel in kernels:
      results.append(pool.apply_async(self._worker, (img, kernel, feat_fn)))
    pool.close()
    pool.join()
    
    hist = np.array([res.get() for res in results])
  
    if normalize:
      hist = hist / np.sum(hist, axis=0)
  
    return hist.T.flatten()
  
  
  def _worker(self, img, kernel, feat_fn):
    try:
      ret = feat_fn(img, kernel)
    except:
      print("return zero")
      ret = np.zeros(2)
    return ret
  
  
  def extract_features(self, db, query_image_filepath=None, is_query=False, verbose=True):
    if h_type == 'global':
      db_img_features_cache = "gabor-{}-theta{}-frequency{}-sigma{}-bandwidth{}".format(h_type, theta, frequency, sigma, bandwidth)
    elif h_type == 'region':
      db_img_features_cache = "gabor-{}-n_slice{}-theta{}-frequency{}-sigma{}-bandwidth{}".format(h_type, n_slice, theta, frequency, sigma, bandwidth)

    # print("Now making Gabor samples.")

    if is_query == False:
      # If extracting features from all images in database
      try:
        # Use cached image features
        db_images = cPickle.load(open(os.path.join(cache_dir, db_img_features_cache), "rb", True))
        for sample in db_images:
          sample['hist'] /= np.sum(sample['hist'])  # normalize
        if verbose:
          print("Using cache..., config=%s, distance=%s, depth=%s" % (db_img_features_cache, d_type, depth))
      except:
        if verbose:
          print("Counting histogram..., config=%s, distance=%s, depth=%s" % (db_img_features_cache, d_type, depth))

        db_images = []
        data = db.get_data()
        for d in data.itertuples():
          d_img, d_cls = getattr(d, "img"), getattr(d, "cls")

          # Generate Gabor histogram for each image in the database
          d_hist = self.gabor_histogram(d_img, type=h_type, n_slice=n_slice)
          db_images.append({
            'img': d_img,
            'cls': d_cls,
            'hist': d_hist
          })
        # Cache extracted image features
        cPickle.dump(db_images, open(os.path.join(cache_dir, db_img_features_cache), "wb", True))
        print("Gabor samples: ", db_images)
      return db_images
    elif is_query == True:
      # If extracting features from single image
      d_hist = self.gabor_histogram(query_image_filepath, type=h_type, n_slice=n_slice)
      query = {
        'img': query_image_filepath,
        'cls': "query",
        'hist': d_hist
      }
      return query
