# encoding=utf-8

import os
import sys
import time
import numpy as np

# For saving trained models
from joblib import dump, load

# sklearn tools
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import GridSearchCV

# Models
# from sklearn.svm import SVR
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

# Custom Regressor Stacker
from stack import Stacker

def load_social_features(video_id, video_user, user_details):
    vid = [] #video id list
    for line in open(video_id):
        vid.append(line.strip())
   
    vid_uid_dict = {} #vid-uid mapping
    for line in open(video_user):
        data = line.strip().split('::::')
        vid_uid_dict[data[0]] = data[1]
    
    social_features = {} #uid-social_feature mapping
    for line in open(user_details, encoding='utf-8'):
        data = line.strip().split("::::")

        """
            # Modified Social Features **
            1. Total Loop Count
            2. Average Loop Count
            3. Follower Count
            4. Follower / Followee Ratio
            social_features[data[0]] = [float(data[1]),
                                        float(data[1]) / float(data[5]),
                                        float(data[2]),
                                        float(data[2]) / float(data[3])]
        """

        # Original Social Features = Total Loops + Follower Count
        social_features[data[0]] = [float(data[1]), float(data[2])]

    res = []
    for v in vid:
        try:
            res.append(social_features[vid_uid_dict[v]])
        except:
            # note: there are some users don't have social features, just assign zero-vector to them
            # update: remove these later on so matrices are not singular
            res.append([0.0, 0.0]) 

    return np.array(res, dtype=np.float32)

def load_text_sent_features(sent_scores):
    """
    Convert sentiment scores into classes of 0, 1, 2
    for negative, neutral, and positive respectively
    """
    with open(sent_scores, encoding='utf-8') as f:
        scores = []
        for x in f.readlines():
            score = float(x)
            if score < -0.3:
                scores += [0]
            elif score <= 0.3:
                scores += [1]
            else:
                scores += [2]
        return np.array(scores).reshape(-1,1)

def main(record):
    data_dir = './data/' 
    
    # load data
    print("Loading data...")

    # load ground-truth
    ground_truth = []
    for line in open(os.path.join(data_dir, 'ground_truth.txt')):
        # you can use more than one popularity index as ground-truth and average the results; for each video we have four indexes: number of loops(view), likes, reposts, and comments; the first one(loops) is compulsory.
        ground_truth.append(float(line.strip().split('::::')[0])) 
    ground_truth = np.array(ground_truth, dtype=np.float32)

    # PCA testing vals
    PCA_vals = {
        'imgNet' : [20, 50, 100, 200, 500, 1000],
        'vSenti' : [40, 100, 200, 500, 1000, 1500, 2000],
        'sen2vec' : [10, 20, 50, 75, 100]
    }

    # Visual
    hist_feature = np.load(data_dir + 'histogram_feature.npz')['arr_0']
    imgNet_feature = PCA(n_components=PCA_vals['imgNet'][0]).fit_transform(np.load(data_dir + 'imageNet_feature.npz')['arr_0'])
    vSenti_feature = PCA(n_components=PCA_vals['vSenti'][0]).fit_transform(np.load(data_dir + 'visual_senti_feature.npz')['arr_0'])
    visual_feature = np.concatenate([hist_feature, imgNet_feature, vSenti_feature], axis=1)

    # Text
    sen2vec_feature = PCA(n_components=PCA_vals['sen2vec'][0]).fit_transform(np.load(data_dir + 'text_sentence2vec_feature.npz')['arr_0'])
    text_sent_feature = load_text_sent_features(data_dir+'text_sentiment.txt')
    text_feature = np.concatenate([sen2vec_feature, text_sent_feature], axis=1)

    # Social
    social_feature = load_social_features(data_dir + 'video_id.txt', data_dir + 'video_user.txt', data_dir + 'user_details.txt')

    # concatenate all the features(after dimension reduction)
    concat_feature = visual_feature
    # concat_feature = np.concatenate([text_feature, visual_feature], axis=1)

    # remove the empty social feature indices
    if (social_feature.shape[1] == concat_feature.shape[1]):
        empty_indices = []
        for i in range(len(social_feature)):
            if np.array_equal(social_feature[i],[0,0]):
                empty_indices += [i]

        concat_feature = np.delete(concat_feature, empty_indices, 0)
        ground_truth = np.delete(ground_truth, empty_indices, 0)
    else:
        # Prepare Features with Percentile -- unless only social modality
        # f_selector = SelectPercentile(f_classif, percentile=70)
        # concat_feature = f_selector.fit_transform(concat_feature, ground_truth)
        pass
    print("The input data dimension is: (%d, %d)" % (concat_feature.shape))
    
    print("Start training and predict...")
    # classifier = SVR(C=30, gamma=0.01)
    # classifier = KernelRidge(alpha=3.0, kernel='rbf')
    # classifier = MLPRegressor(max_iter=200)
    # classifier = AdaBoostRegressor()
    classifier = BaggingRegressor()


    kf = KFold(n_splits=10)
    nMSEs = []
    count = 0

    for train, test in kf.split(concat_feature):

        """
            # Late Fusion --> for more info: look at stack.py
            vis_class, text_class, social_class = Stacker(classifier), Stacker(classifier), Stacker(classifier)
            
            # Update feature vector
            x = np.zeros((len(concat_feature), 3)) 
            x[train, 0] = vis_class.fit_transform(visual_feature[train,:], ground_truth[train])[:,0]
            x[test, 0] = vis_class.transform(visual_feature[test,:])
            x[train, 1] = text_class.fit_transform(text_feature[train,:], ground_truth[train])[:,0]
            x[test, 1] = text_class.transform(text_feature[test])
            x[train, 2] = social_class.fit_transform(social_feature[train,:], ground_truth[train])[:,0]
            x[test, 2] = social_class.transform(social_feature[test,:])

            model = classifier.fit(x[train,:], ground_truth[train])
            predicts = model.predict(x[test,:])
        """

        # train
        model = classifier.fit(concat_feature[train], ground_truth[train])
        
        # predict
        predicts = model.predict(concat_feature[test])

        nMSE = mean_squared_error(ground_truth[test], predicts) / np.mean(np.square(ground_truth[test]))
        nMSEs.append(nMSE)

        count += 1
        print("Round %d/10 of nMSE is: %f" %(count, nMSE))
    
    print('Average nMSE is %f' %(np.mean(nMSEs)))

    # Optionally record and look at Results
    if record:
        NUM_RECORDS = 100
        res_file = "res_%d.txt" % int(time.time())
        with open(os.path.join(res_file), 'w') as f:
            for i in range(NUM_RECORDS):
                f.write('%s %s\n' % (str(ground_truth[i]), str(model.predict(concat_feature[i].reshape(1,-1))[0])))

if __name__ == "__main__":
    record = False

    # Takes in a parameter to record results into a file.
    if len(sys.argv) > 1:
        record = sys.argv[1] == 't' or sys.argv[1] == 'true'

    main(record)
