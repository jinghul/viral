# Viral
Predict the popularity of VINE videos through multi-modal feature union.

### Dataset
For this section, please refer to data/README.md


### Environment Setting
1. Ubuntu 16.04
2. Python 2.7


### Installation 
1. numpy
2. scikit-learn
Note: All these libraries can be installed via pip. (e.g., pip install nltk)  


### Usage 
1. Run 'python basic_svr_for_viral_prediction.py' to run a baseline regression model of SVR for viral item prediction.

### Evaluation metric
1. We use nMSE(normalized Nean Squared Error) to evaluate the performance of the prediction model. For more details about nMSE, you can refer to Equation 22 in section 6.1 of [2].
2. You are required to run 10-fold cross-validation. For the popularity indexes, the count of loop is compulsory to use, while you can evaluate on more than one popularity index.

### Comments
There are detailed comments within the code: basic_svr_for_viral_prediction.py, which can explain itself.

### References
[1] J. Chen. Multi-modal learning: Study on A large-scale micro-video data collection.  In Proceedings of the 2016 ACM Conference on Multimedia Conference, MM 2016, Amsterdam, Netherlands, October 15-19, 2016, pages 1454–1458. ACM, 2016.
[2] J. Chen, X. Song, L. Nie, X. Wang, H. Zhang, and T. Chua. Micro tells macro: Predicting the popularity of micro-videos via a transductive model. In MM, pages 898–907. ACM, 2016
