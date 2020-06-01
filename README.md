CS 7641: Machine Learning Model Analysis
===

This project was done as a part of CS7641 Coursework. 5 machine learning models
-- Decision Boosting, Ensemble Boosting, K-Nearest Neighbors, Neural Networks,
and Support Vector Machines -- were fitted against gesture data and pulsar data
from Kaggle. The objective was to measure the performance of the different 
models and do a comparison.


To Run Analysis
---

1. Install miniconda (follow instructions here: https://docs.conda.io/en/latest/miniconda.html)
2. Create the environment in terminal using the below command. Make sure that your current directory in terminal is the root folder for this project:
    i. conda env create -f environment.yml
3. Activate the environment with below command:
    i. conda activate cs7641_assignment1
4. Select a data source by (un)commenting the appropriate line in script.py
    i. For the pulsar data, uncomment line 14 and comment line 15
    ii. For gesture data, uncomment line 15
5. Select a classifier to analyze by adding the appropriate method call at the end of script.py
    i. Decision Tree Analysis: decision_tree()
    ii. Neural networks: neural_networks()
    iii. K-Nearest Neighbors: knn()
    iv. Boosting: boosting()
    v. Support Vector Machines: svc()
6. Results will be outputed to ./out/ folder
