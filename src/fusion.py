# -*- coding: utf-8 -*-

from __future__ import print_function

# from evaluate import extract_features_from_db
# from DB import Database
from src.evaluate import extract_features_from_db
from src.DB import Database

# from color import Color
# from edge  import Edge
# from gabor import Gabor

from src.color import Color
from src.edge  import Edge
from src.gabor import Gabor


import numpy as np
import itertools
import os


d_type   = 'd1'
depth    = 30

feat_pools = ['color', 'edge', 'gabor']

# result dir
result_dir = 'result'
if not os.path.exists(result_dir):
  os.makedirs(result_dir)


class FeatureFusion(object):

  def __init__(self, features):
    assert len(features) > 1, "need to fuse more than one feature!"
    self.features = features
    self.images  = None

  def extract_features(self, db, verbose=False):
    if verbose:
      print("Use features {}".format(" & ".join(self.features)))

    if self.images == None:
      feats = []
      for f_class in self.features:
        feats.append(self._get_feat(db, f_class))
      # print("feats: ", feats)
      images = self._concat_feat(db, feats)
      # print("samples: ", images)
      self.images = images  # cache the result
    return self.images

  # Extract multiple features from query image
  def extract_features_from_query_img(self, query_image_filepath):
    feats = []
    for f_class in self.features:
      feats.append(self.extract_query_features(query_image_filepath, f_class))
    # print("feats: ", feats)
    images = self.concat_query_features(feats)
    return images

  # Extract one type of feature from query image
  def extract_query_features(self, query_image_filepath, f_class):
    if f_class == 'color':
      f_c = Color()
    elif f_class == 'edge':
      f_c = Edge()
    elif f_class == 'gabor':
      f_c = Gabor()
    return f_c.extract_features(db=None, query_image_filepath=query_image_filepath, is_query=True, verbose=False)

  # Extract one type of feature from all images in database
  def _get_feat(self, db, f_class):
    if f_class == 'color':
      f_c = Color()
    elif f_class == 'edge':
      f_c = Edge()
    elif f_class == 'gabor':
      f_c = Gabor()
    return f_c.extract_features(db, verbose=False)

  # Concatenate image features for the query image
  def concat_query_features(self, feats):
    first_item = feats[0]
    for i in feats[1:]:
      first_item['hist'] = np.append(first_item['hist'], i["hist"])
    return first_item

  # Concatenate image features for each image in the database
  def _concat_feat(self, db, feats):
    images = feats[0]
    delete_idx = []
    for idx in range(len(images)):
      for feat in feats[1:]:
        feat = self._to_dict(feat)
        key = images[idx]['img']
        if key not in feat:
          delete_idx.append(idx)
          continue
        assert feat[key]['cls'] == images[idx]['cls']
        images[idx]['hist'] = np.append(images[idx]['hist'], feat[key]['hist'])
    for d_idx in sorted(set(delete_idx), reverse=True):
      del images[d_idx]
    if delete_idx != []:
      print("Ignore %d samples" % len(set(delete_idx)))

    return images

  # Convert image information to dictionary
  def _to_dict(self, feat):
    ret = {}
    for f in feat:
      ret[f['img']] = {
        'cls': f['cls'],
        'hist': f['hist']
      }
    return ret


# Measure the performance of every combination of image features
def evaluate_feats(db, N, feat_pools=feat_pools, d_type='d1', depths=[None, 300, 200, 100, 50, 30, 10, 5, 3, 1]):
  result = open(os.path.join(result_dir, 'feature_fusion-{}-{}feats.csv'.format(d_type, N)), 'w')
  for i in range(N):
    result.write("feat{},".format(i))
  result.write("depth,distance,MMAP")
  combinations = itertools.combinations(feat_pools, N)
  for combination in combinations:
    fusion = FeatureFusion(features=list(combination))
    for d in depths:
      APs = extract_features_from_db(db, feature_instance=fusion, d_type=d_type, depth=d)
      cls_MAPs = []
      for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        cls_MAPs.append(MAP)
      r = "{},{},{},{}".format(",".join(combination), d, d_type, np.mean(cls_MAPs))
      print(r)
      result.write('\n'+r)
    print()
  result.close()



