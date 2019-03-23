# encoding=utf-8

import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


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

        # You should modify here to add more user social information
        # Here we only use two user social infomation: loops and followers. You should consider more user social information. For more details about other social information, pls refer to ./data/README.txt -> 4.user_details.txt 
        social_features[data[0]] = [float(i) for i in data[1:3]] 

    res = [] #social_feature vector for each video
    for v in vid:
        try:
            res.append(social_features[vid_uid_dict[v]])
        except:
            #note: there are some users don't have social features, just assgin zero-vector to them
            res.append([0.0, 0.0]) 

    return np.array(res, dtype=np.float32) 

def load_text_sent_features(sent_scores):
    with open(sent_scores, encoding='utf-8') as f:
        scores = f.readlines()
        return np.array(scores).reshape(-1,1)

def main():
    data_dir = './data/' 
    
    # load data
    print("Loading data...")

    # Visual
    hist_feature = np.load(data_dir + 'histogram_feature.npz')['arr_0']
    imgNet_feature = np.load(data_dir + 'imageNet_feature.npz')['arr_0']
    vSenti_feature = np.load(data_dir + 'visual_senti_feature.npz')['arr_0']

    # Text
    sen2vec_feature = np.load(data_dir + 'text_sentence2vec_feature.npz')['arr_0']
    text_sent_feature = load_text_sent_features(data_dir+'text_sentiment.txt')

    # Social
    social_feature = load_social_features(data_dir + 'video_id.txt', data_dir + 'video_user.txt', data_dir + 'user_details.txt')

    # feature dimension reduction: it's up to you to decide the size of reduced dimensions; the main purpose is to reduce the computation complexity
    pca = PCA(n_components=20)
    imgNet_feature = pca.fit_transform(imgNet_feature)
    pca = PCA(n_components=40)
    vSenti_feature = pca.fit_transform(vSenti_feature)
    pca = PCA(n_components=10)
    sen2vec_feature = pca.fit_transform(sen2vec_feature)
    
    # contatenate all the features(after dimension reduction)
    concat_feature = np.concatenate([hist_feature, imgNet_feature, vSenti_feature, sen2vec_feature, text_sent_feature, social_feature], axis=1) 
    print("The input data dimension is: (%d, %d)" %(concat_feature.shape))
    
    # load ground-truth
    ground_truth = []
    for line in open(os.path.join(data_dir, 'ground_truth.txt')):
        # you can use more than one popularity index as ground-truth and average the results; for each video we have four indexes: number of loops(view), likes, reposts, and comments; the first one(loops) is compulsory.
        ground_truth.append(float(line.strip().split('::::')[0])) 
    ground_truth = np.array(ground_truth, dtype=np.float32)
    
    
    print("Start training and predict...")
    classifier = SVR(gamma='auto')
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
        print("Round %f/10 of nMSE is: %f" %(count, nMSE))
    
    print('Average nMSE is %f' %(np.mean(nMSEs)))


if __name__ == "__main__":
    main()
