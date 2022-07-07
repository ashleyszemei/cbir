from flask import Flask, request, render_template
from six.moves import cPickle
import os
import shutil
import numpy as np

from src.evaluate import infer, extract_features_from_db
from src.DB import Database
from src.color import Color
from src.edge import Edge
from src.gabor import Gabor
from src.fusion import FeatureFusion

# This is the texture branch

app = Flask(__name__)

UPLOAD_FOLDER = "static/upload"

# Distance type
d_type  = 'd1'




@app.route("/", methods=["GET", "POST"])
def home():
    # After image is uploaded
    if request.method == "POST":
        # Clear contents of upload folder
        shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)

        # Get the uploaded image
        image_file = request.files["query_image"]

        # Get selected feature extraction method
        selected_features = request.form.getlist("image_feature")
        print(selected_features)
        # ['Color', 'Gabor', 'Edge']

        # Check if image is .jpg file, then save to upload folder
        if image_file.filename != '':
            if image_file.filename.lower().endswith('.jpg'):
                query_image_filepath = os.path.join(UPLOAD_FOLDER, image_file.filename)
                image_file.save(query_image_filepath)

        db = Database()

        if len(selected_features) == 1:
            # If only one feature is selected
            feature_type = selected_features[0]

            # Get the selected feature
            if feature_type == "Color":
                feature_class = Color
            elif feature_type == "Gabor":
                feature_class = Gabor
            elif feature_type == "Edge":
                feature_class = Edge
            f = feature_class()

            # Extract features from query image
            query = f.extract_features(db=None, query_image_filepath=query_image_filepath, is_query=True)

            # Extract features from all images in the database
            APs, samples = extract_features_from_db(db, f_class=feature_class, d_type=d_type)
            cls_MAPs = []
            for cls, cls_APs in APs.items():
                MAP = np.mean(cls_APs)
                print("Class {}, MAP {}".format(cls, MAP))
                cls_MAPs.append(MAP)

            # Calculate the MMAP of feature extraction from database images
            MMAP = np.mean(cls_MAPs)
        elif len(selected_features) > 1:
            # If more than one feature is selected
            # Convert list of image features to lowercase
            lowercase_features = [x.lower() for x in selected_features]

            # Extract multiple features from query image
            fusion = FeatureFusion(features=lowercase_features)
            query = fusion.extract_features_from_query_img(query_image_filepath)

            # Extract multiple features from all images in the database
            APs, samples = extract_features_from_db(db, f_instance=fusion, d_type=d_type)
            cls_MAPs = []
            for cls, cls_APs in APs.items():
                MAP = np.mean(cls_APs)
                print("Class {}, MAP {}".format(cls, MAP))
                cls_MAPs.append(MAP)

            # Calculate the MMAP of feature extraction from database images
            MMAP = np.mean(cls_MAPs)

        # Calculate the similarity between query image and database images, and return the 5 most similar images
        _, results = infer(query, samples, db=None, sample_db_fn=None, depth=5, d_type='d1')

        # Make a copy of results
        image_results = results.copy()

        # Change the formatting of the image file path so that it can be read by Flask application
        for image_result in image_results:
            img_url = image_result["img"]
            flask_url = img_url.replace(os.sep, '/')
            image_result["img"] = flask_url

        return render_template(
            "index.html",
            query_image_filepath=query_image_filepath,
            image_results=image_results,
            MMAP=MMAP
        )
    else:
        # Clear contents of upload folder
        shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)

        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)