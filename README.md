# Reddit Score Predictor

This project aims to find out what makes a Reddit post popular and how well we can predict the score of a Reddit post using machine learning. This project consists of three main components: the data processing pipeline (Pipeline), analysis scripts (Analysis), and stored figures (Figures). Additionally, there are two models, **predict_score_old.py** and **predict_score_new.ipynb**.

## Project Overview
This project is organized into three main sections:

**/pipeline**
- The data processing pipeline responsible for cleaning, transforming, and preparing the Reddit data for analysis and modeling

**/analysis**
- An investigation into the Reddit Posts dataset, our processed data, and the inital **predict_score_old.py** results. This section provides insights into the characteristics of Reddit posts and their scores.

**/figures**
- Visualizations from **3-initial_analysis.py** used in our analysis

## Approaches

### Old Predictor
**predict_score-old.py** is the inital approach to predict the scores of Reddit posts. This was done using Spark's Linear Regression and was compared against a dummy regressor using the mean.

### New Predictor
**predict_score_new.ipynb** is the revised approach to predict the scores of Reddit posts and analysis on its accuracy. This was done using Linear Regression, Random Forest Regression, KNN Regression, and Decision Tree Regression from Scikit-Learn. This approach also used both the mean and median as dummy regressors. This approach was done after reviewing the results from **predict_score_old.py**.

## Next Steps
The aim of **predict_score_new.ipynb** was to improve the accuracy of **predict_score_old.py**, which it did. However, certain planned improvements, such as generating new features using semantic analysis and word embeddings, were hindered by a reduction in compute availability that we previously had access to.

If we gain access to large compute power, we would also like to:
- Generate new features using semantic analysis and word embeddings
- Train a neural network and compare its results to the models we've already tested
- Apply undersampling to posts with a low score as our dataset is heavily right skewed
- Apply feature selection to improve the accuracy of our models
- Apply hyperparameter tuning to all of our models