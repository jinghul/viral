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
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

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

        # Social Features **
        # 1. Total Loop Count
        # 2. Average Loop Count
        # 3. Average Like Count
        # 4. Follower Count
        # 5. Follower / Followee Ratio
        social_features[data[0]] = [float(data[1]), \
                                    float(data[1]) / float(data[5]), \
                                    float(data[4]) / float(data[5]), \
                                    float(data[2]), \
                                    float(data[2]) / float(data[3])]

    res = [] #social_feature vector for each video
    for v in vid:
        try:
            res.append(social_features[vid_uid_dict[v]])
        except:
            #note: there are some users don't have social features, just assgin zero-vector to them
            res.append([0.0, 0.0, 0.0, 0.0, 0.0]) 

    return np.array(res, dtype=np.float32)

def load_text_sent_features(sent_scores):
    with open(sent_scores, encoding='utf-8') as f:
        scores = []
        for x in f.readlines():
            score = float(x)
            if score < -0.25:
                scores += [0]
            elif score <= 0.25:
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

    # Visual
    hist_feature = np.load(data_dir + 'histogram_feature.npz')['arr_0']
    imgNet_feature = PCA(n_components=20).fit_transform(np.load(data_dir + 'imageNet_feature.npz')['arr_0'])
    vSenti_feature = PCA(n_components=40).fit_transform(np.load(data_dir + 'visual_senti_feature.npz')['arr_0'])
    visual_feature = np.concatenate([hist_feature, imgNet_feature, vSenti_feature], axis=1)

    # Text
    sen2vec_feature = PCA(n_components=10).fit_transform(np.load(data_dir + 'text_sentence2vec_feature.npz')['arr_0'])
    text_sent_feature = load_text_sent_features(data_dir+'text_sentiment.txt')
    text_feature = np.concatenate([sen2vec_feature, text_sent_feature], axis=1)

    # Social
    social_feature = load_social_features(data_dir + 'video_id.txt', data_dir + 'video_user.txt', data_dir + 'user_details.txt')

    # concatenate all the features(after dimension reduction)
    # concat_feature = social_feature
    concat_feature = np.concatenate([visual_feature, social_feature, text_feature], axis=1) 
    
    # Prepare Features with Percentile
    # f_selector = SelectPercentile(f_classif, percentile=70)
    # concat_feature = f_selector.fit_transform(concat_feature, ground_truth)
    print("The input data dimension is: (%d, %d)" % (concat_feature.shape))
    
    print("Start training and predict...")
    # classifier = SVR(gamma='auto')
    classifier = KernelRidge(alpha=1.0, kernel='rbf')

    kf = KFold(n_splits=10)
    nMSEs = []
    count = 0

    for train, test in kf.split(concat_feature):

        # train
        model = classifier.fit(concat_feature[train], ground_truth[train])
        
        # predict
        predicts = model.predict(concat_feature[test])

        nMSE = mean_squared_error(ground_truth[test], predicts) / np.mean(np.square(ground_truth[test]))
        nMSEs.append(nMSE)

        count += 1
        print("Round %d/10 of nMSE is: %f" %(count, nMSE))

        
    
    print('Average nMSE is %f' %(np.mean(nMSEs)))

    # Look at Results
    if record:
        res_file = "res_%d" % int(time.time())
        with open(os.path.join(res_file), 'w') as f:
            for i in range(100):
                f.write('%s %s %s\n' % (str(concat_feature[i]), str(ground_truth[i]), str(model.predict(concat_feature[i].reshape(1,-1))[0])))

if __name__ == "__main__":
    record = False
    if len(sys.argv) > 1:
        record = sys.argv[1] == 't' or sys.argv[1] == 'true'
    main(record)
