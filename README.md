# Reddit Score Predictor

This project aims to find out what makes a Reddit post popular and how well we can predict the score of a Reddit post using machine learning. This project consists of three main components: the data processing pipeline (/pipeline), analysis scripts (/analysis), and the score predictors.

## Project Overview
This project is organized into four main sections:

**/pipeline**: the data processing pipeline responsible for cleaning, transforming, and preparing the Reddit data for analysis and modeling

**/analysis**: an investigation into the Reddit Posts dataset, our processed data, and the inital **predict_score_old.py** results. This section provides insights into the characteristics of Reddit posts and their scores.

**/figures**: visualizations from **3-initial_analysis.py** used in our analysis

There is also two score predictors **predict_score_old.py** and **predict_score_new.ipynb**, as well as **report.pdf** which contains a written report of this project excluding the **predict_score_new.ipynb** portion.

## Getting Started
1. Installing requirements
```sh
$ pip install -r  requirements.txt
```

2. Running the pipeline

The data processing pipeline was originally created on a remote cluster that utilized the HDFS, so the pathnames in these files may not be correct. The datasets used in this project can be found here: https://github.com/webis-de/webis-tldr-17-corpus.

Run each file from lowest starting number to highest using:
```sh
$ spark-submit \#-filename.py
```

3. Data analysis
The data analysis must run after the data processing pipeline, and **visualized_model_erro.ipynb** must be run after **predict_score_old.py**.

4. Predictors
Both **predict_score_old.py** and **predict_score_new.ipynb** must be run after the data processing pipeline

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