from __future__ import print_function

from skimage.feature import greycomatrix, greycoprops
from skimage import color
from six.moves import cPickle
import numpy as np
import os
import imageio
import cv2

from src.evaluate import distance, extract_features_from_db
from src.DB import Database

n_slice = 4
d_type  = 'd1'

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)

def calc_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], lvl=256, sym=True,
                       norm=True):
    glcm = greycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)

    return feature

class Glcm(object):
    def glcm_matrix(self, input):
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')

        height, width, channel = img.shape

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ymin, ymax, xmin, xmax = height // 3, height * 2 // 3, width // 3, width * 2 // 3
        crop = gray_img[ymin:ymax, xmin:xmax]
        resize = cv2.resize(crop, (0, 0), fx=0.5, fy=0.5)

        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
        glcm_all_agls = calc_glcm_all_agls(resize, props=properties)
        glcm_all_agls = np.array(glcm_all_agls)
        print(glcm_all_agls)

        return glcm_all_agls

    def extract_features(self, db, query_image_filepath=None, is_query=False, verbose=True):
        db_img_features_cache = "glcm"
        if is_query == False:
            # If extracting features from all images in database
            try:
                # Use cached image features
                print(db_img_features_cache)
                db_images = cPickle.load(open(os.path.join(cache_dir, db_img_features_cache), "rb", True))
                if verbose:
                    print("Using cache...")
            except:
                if verbose:
                    print("Counting matrix...")

                db_images = []
                data = db.get_data()
                for d in data.itertuples():
                    d_img, d_cls = getattr(d, "img"), getattr(d, "cls")

                    d_hist = self.glcm_matrix(d_img)
                    db_images.append({
                        'img': d_img,
                        'cls': d_cls,
                        'hist': d_hist
                    })
                # Cache extracted image features
                cPickle.dump(db_images, open(os.path.join(cache_dir, db_img_features_cache), "wb", True))
            return db_images
        elif is_query == True:
            # If extracting features from single image
            d_hist = self.glcm_matrix(query_image_filepath)
            query = {
                'img': query_image_filepath,
                'cls': "query",
                'hist': d_hist
            }
            return query

if __name__ == "__main__":
  db = Database()
  data = db.get_data()
  print(data)
  graylevel = Glcm()

  # evaluate database
  APs = extract_features_from_db(db, feature_class=Glcm, d_type=d_type, depth=None)
  print(APs)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
      MAP = np.mean(cls_APs)
      print("Class {}, MAP {}".format(cls, MAP))
      cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))





# input = 'C:/Users/User/PycharmProjects/cbirTest/query_canai.jpg'
# img = imageio.imread(input, pilmode='RGB')
# height, width, channel = img.shape
#
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ymin, ymax, xmin, xmax = height//3, height*2//3, width//3, width*2//3
# crop = gray_img[ymin:ymax, xmin:xmax]
# resize = cv2.resize(crop, (0,0), fx=0.5, fy=0.5)
#
# properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
# glcm_all_angles = []
# glcm_all_agls = calc_glcm_all_agls(resize, props=properties)
# print(glcm_all_agls)
# print(len(glcm_all_agls))


# cv2.imshow('gray', resize)
# cv2.waitKey(0)   #wait for a keyboard input
# cv2.destroyAllWindows()












