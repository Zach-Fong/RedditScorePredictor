# Reddit Score Predictor

This project aims to explore makes a Reddit post popular and how well we can predict the score of a Reddit post using machine learning.

## Project Overview
This project is organized into these main sections:

**report.pdf**: contains a report of all pipeline, analysis, and modelling processes (excluding **predict_score_new.ipynb**).

**/pipeline**: the data processing pipeline responsible for cleaning, transforming, and preparing the Reddit data for analysis and modeling.

**/analysis**: an investigation into the Reddit Posts dataset, our processed data, and the inital **predict_score_old.py** results.

**/models**: models that were used to predict a posts score. This includes an intial approach (**predict_score_old.py**) and a revised approach (**predict_score_new.ipynb**).

**/figures**: visualizations from **3-initial_analysis.py** used in our analysis.

## Getting Started
1. Installing requirements
```sh
$ pip install -r  requirements.txt
```

2. Running the pipeline

The data processing pipeline was originally created on a remote cluster that utilized the HDFS, so the pathnames in these files may not be applicable. The datasets used in this project can be found here: https://github.com/webis-de/webis-tldr-17-corpus.

Run each file from the lowest starting number to highest using:
```sh
$ spark-submit #-filename.py
```

3. Data analysis

Data analysis must run after the data processing pipeline, and **visualized_model_error.ipynb** must be run after **predict_score_old.py**.

4. Predictors

Both **predict_score_old.py** and **predict_score_new.ipynb** must be run after the data processing pipeline

## Approaches

### Old Predictor
**predict_score-old.py** is the inital approach to predict the scores of Reddit posts. This was done using Spark's Linear Regression and was compared against a mean dummy regressor.

### New Predictor
**predict_score_new.ipynb** is the revised approach to predict the scores of Reddit posts and contains analysis on its accuracy. This was done using Linear Regression, Random Forest Regression, KNN Regression, and Decision Tree Regression from Scikit-Learn. This approach also uses both the mean and median as dummy regressors. This approach was done after reviewing the results from **predict_score_old.py**.

## Next Steps
The aim of **predict_score_new.ipynb** was to improve the accuracy of **predict_score_old.py**, which it did. However, certain planned improvements, such as generating new features using semantic analysis and word embeddings, were hindered by a reduction in compute availability that we previously had access to.

If we gain access to large compute power, we would also like to:
- Generate new features using semantic analysis and word embeddings
- Train a neural network and compare its results to the models we've already tested
- Apply undersampling to posts with a low score as our dataset is heavily right skewed
- Apply feature selection to improve the accuracy of our models
- Apply hyperparameter tuning to all of our models