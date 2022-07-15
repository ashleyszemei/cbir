from __future__ import print_function

from skimage.feature import greycomatrix, greycoprops
from skimage import color
from six.moves import cPickle
import numpy as np
import os
import imageio
import cv2
from PIL import Image

n_slice = 4
d_type  = 'd1'

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)

def calculate_glcm_properties(img, props, distances=[5], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True,
                       normed=True):
    # Calculate the gray-level co-occurrence matrix
    glcm = greycomatrix(img,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=symmetric,
                        normed=normed)
    feature = []
    # Calculate texture properties from the GLCM
    glcm_properties = [property for name in props for property in greycoprops(glcm, name)[0]]
    for item in glcm_properties:
        feature.append(item)

    return feature

class Glcm(object):
    def calculate_glcm(self, input):
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = imageio.imread(input, pilmode='RGB')

        # Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize image to square
        resize = cv2.resize(gray_img, (224, 224), interpolation=cv2.INTER_AREA)

        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
        cooccurrencematrix = calculate_glcm_properties(resize, props=properties)
        cooccurrencematrix = np.array(cooccurrencematrix)

        return cooccurrencematrix

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

                    d_hist = self.calculate_glcm(d_img)
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
            d_hist = self.calculate_glcm(query_image_filepath)
            query = {
                'img': query_image_filepath,
                'cls': "query",
                'hist': d_hist
            }
            return query

















