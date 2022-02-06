#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Fall 99 <img src = 'https://ece.ut.ac.ir/cict-theme/images/footer-logo.png' alt="Tehran-University-Logo" width="150" height="150" align="right">
# ## Final Project : Film Genres Classification for Cafebazaar
# ### Dr. Abolghasemi and Dr. Araabi
# ### By Omid Vaheb and Mahsa Massoud

# ## Introduction:
# In this project, after inspecting data, we prepared and normalized it for implementing learning algorithms on it. The most significant barrier in the way was doing preprocessing for texts in persian but we handled it with Hazm library. The final step was to enhance and set hyperparameters for some classification models to get the maximum accuracy and minimum MSE for each model.
# ## Question:
# In order to predict the genre of a film in Cafebazaar's database, we need to build a model using machine learning algorithms. The dataset we used in this project was the real data of films gathered by a company. First we train our models using the given dataset. This dataset consists of video_id(which shows the id of film in database and probably would ot help us in classification), title_fa(title of films in persian which could be helpful),  description_fa(description of films which is the main feature to classify with), age_rating(minimum age to watch the movie), country_fa(showing the country that movie vas produced in), production_year(year of production), duration(showing duration of film), image(showing name of the file containing film's poster, and finaly genre which is the main feature to estimate.

# Before anything we import libraries needed in the project. These libraries consist of 3 different groups. First one is the general libraries like pandas and numpy and some learning algorithms which are mostly included in sklearn and scipy. Second group is related to the text processing for example hazm, nltk, and re. The last group are added in orer to do image processing for posters. Generally opencv fulfills our needs of image processing tools.

# In[177]:


from __future__ import unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import math
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import itertools
import gc
get_ipython().system('pip install hazm')
import hazm
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
get_ipython().system('pip install lightgbm')
import random
import lightgbm as lgb
from lightgbm import LGBMRegressor
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# Then, it is time to import the dataset and inspect it using some simple pandas commands. In this project, we uploded csv files into notebook and imported images from a shortcut link to given google drive link.

# In[178]:


from google.colab import files
uploaded = files.upload()


# We sxplore the first ten rows of this dataset to get familiar with its content.

# In[179]:


import io
data = pd.read_csv(io.BytesIO(uploaded['train_set.csv']))
data.head(10)


# A noteworthy observation is that every cell of this dataset have the format of object so we need a transfer for each one when we are processing it.

# In[180]:


data.info()


# We checked the statistical features of dataset in this part using describe.

# In[181]:


data.describe()


# It is obvious that video_id is not a helpful feature for us and we cannot extract usuful information from it like description and title so we drop this column.

# In[182]:


data = data.drop(columns = ['video_id'])
data.head(10)


# There are some NaN values in this dataset and there are also some cells with space instead of value so we need to replace them with NaN. The next step would be to handle these NaN values using the best method available regarding each case.

# In[183]:


data.loc[data.age_rating == " "]


# In[184]:


data.loc[data.production_year == " "]


# In[185]:


data.loc[data.duration == " "]


# There are 118 rows with NaN value in country_fa column so we can asily drop these rows without losing much information because they are about 1% of size of dataset.

# In[186]:


dataframe = data[data['country_fa'].notna()]


# Now we replace cells with space instead of value with NaN.

# In[187]:


dataframe = dataframe.replace(r'^\s*$', np.NaN, regex=True)


# In[188]:


dataframe.isnull().sum()


# There are only 2 rows with missing production year so we drop these 2.

# In[189]:


dataframe = dataframe[dataframe['production_year'].notna()]


# Number of missing data for age rating are more than 2000 and we fill these nan values with the mean age restriction of each class which is a prior knowledge for us. There is a barrier in which drama movies do not have any age rating sample so we have to fill it with the meano of age restriction of whole dataset in which is equal to 13.

# In[190]:


dataframe['age_rating'] = dataframe['age_rating'].astype(float)
data.loc[(data.genre == "drama")]


# In[191]:


dataframe.groupby('genre')['age_rating'].mean()


# In[192]:


dataframe['age_rating'] = dataframe.groupby('genre')['age_rating'].transform(lambda x: x.fillna(x.mean()))


# In[193]:


dataframe['age_rating'].fillna(dataframe['age_rating'].mean(), inplace = True)
dataframe.groupby('genre')['age_rating'].mean()


# In[194]:


dataframe['age_rating'] = dataframe['age_rating'].astype(int)


# In[195]:


dataframe['duration'] = pd.to_datetime(dataframe['duration'])


# Now we try to handle the duration column. First we convert it to datetime and then we break it to hour and minute and finally we aggregate these 2 into the duration bu minute.

# In[196]:


dataframe['hour'] = dataframe['duration'].dt.hour
dataframe['minute'] = dataframe['duration'].dt.minute


# In[197]:


def dateHandler(row): 
    row['minute'] += row['hour'] * 60
    return row
dataframe = dataframe.apply(dateHandler, axis = 'columns')
dataframe = dataframe.drop(columns = ['hour', 'duration'])


# In[198]:


dataframe['minute'] = dataframe['minute'].replace(0, np.NaN)


# Now we fill the nan values of this column with mean of the whole column.

# In[199]:


dataframe['minute'] = dataframe['minute'].fillna(dataframe['minute'].mean())
dataframe.isnull().sum()


# In[200]:


dataframe['production_year'] = dataframe['production_year'].astype(float)
dataframe['production_year'] = dataframe['production_year'].astype(int)
dataframe['minute'] = dataframe['minute'].astype(int)


# In[201]:


dataframe.head(20)


# Now we try to visualize the data available in the dataset using barplots. It can be seen that except romance which is half of other genres and action which has double, others have hardly the same number of data.

# In[202]:


ax = dataframe['genre'].value_counts().plot(kind = 'bar', figsize = (15, 10), title = "Number of Films in Each Genre");
ax.set_xlabel("Genre")
ax.set_ylabel("Count")


# It is obvious that 13 and 17 are more common age restrictions than the others

# In[203]:


ax = dataframe['age_rating'].value_counts().plot(kind = 'bar', figsize = (15, 10), title = "Number of Films with each Age Restriction");
ax.set_xlabel("Age Rating")
ax.set_ylabel("Count")


# Unforunately this part of dataset is not clean at all and we can see that many values are the same but because of mistypes this problem has emerged.

# In[204]:


ax = dataframe['country_fa'].value_counts().plot(kind = 'bar', figsize = (15, 10),  title = "Number of Films Regarding Country of Production");
ax.set_xlabel("Country")
ax.set_ylabel("Count")


# In the next part, I drew barplots of number of films of echa genre in a specific year. It can be interpreted that there is not much of a difference available in this column since these histograms look similar in each year so we decided that production year is not a helpful feature for us.

# In[205]:


for i in range(2000, 2021):
  print("Year :", i)
  ax = dataframe.loc[dataframe['production_year'] == i]['genre'].value_counts().plot(kind = 'bar', figsize = (15, 10), title = "Number of Films in Each Genre in the Specified Year");
  ax.set_xlabel("Genre")
  ax.set_ylabel("Count")
  plt.show()


# In[206]:


dataframe = dataframe.drop(columns = ['production_year'])


# In[207]:


dataframe.head(10)


# Now we vectorize country of production using one hot encoding. The class labels are categorical in nature and have to be converted into numerical form before classification is performed. One-hot encoding is adopted, which converts categorical labels into a vector of binary values. In this method you put a column for each possible value for feature and the value of that column is 1 if that feature is equal to the column's value and zero if not. In this dataset country has 80 values so one hot encoding is a suitable option.

# In[208]:


dataframe = pd.concat([dataframe, pd.get_dummies(dataframe['country_fa'], prefix = 'country')], axis = 1)
dataaa = dataframe
dataframe = dataframe.drop(columns = ['country_fa'])


# In[209]:


dataframe.head(10)


# Unfortunately there is no solution for handling missing values in description column so we leave untouched. The final step of preprocessing is to process 2 remaining unchanged columns,Title and description, and extrct feature vector from them:
# These columns contain some persian texts and they also have some few english words and characters so by using hazm and nltk libraries we cleaned them. The actions taken were normalization, tokenizing and lemmatizing which is a more thorough and accurate method than stemming. Lemmatisation in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. I also created my own set of stopwords regarding dataset.

# In[210]:


def preprocessTextofColumn(columnName):
    normalizer = hazm.Normalizer()
    tokenizer = hazm.WordTokenizer()
    lemmatizer = hazm.Lemmatizer()
    ENGStopWords = set(stopwords.words('english'))
    PERStopWords = set(hazm.utils.stopwords_list())
    PERStopWords = PERStopWords.union({':','ی','ای',',','،','(',')',':',';','-','_','.','/','+','=','?'})
    stopWords = ENGStopWords.union(PERStopWords)
    allWords = []
    for index, row in dataframe.iterrows():
        text = row['title_fa'] + ' ' + row[columnName]
        normalizedText = normalizer.affix_spacing(text)
        words = tokenizer.tokenize(normalizedText)
        filteredWords = []
        for word in words:
            if not word in stopWords:
                filteredWords.append(lemmatizer.lemmatize(word.lower()))
        allWords.append(filteredWords)
    newColumnName = 'Words'
    dataframe[newColumnName] = allWords
    return dataframe

dataframe['description_fa'] = dataframe['description_fa'].astype(str)
dataframe['title_fa'] = dataframe['title_fa'].astype(str)


# In[211]:


import nltk
nltk.download('stopwords')


# In[212]:


dataframe = preprocessTextofColumn('description_fa')


# In[213]:


dataframe = dataframe.drop(columns = ['description_fa', 'title_fa'])


# In[214]:


dataframe.head()


# Now we split the dataset into train and test using sklearn's command train_test_split. It is splited into 80% train and 20% test.

# In[215]:


dataframe2 = dataframe


# In[216]:


X = dataframe.drop(columns = ['genre'])
y = dataframe['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
print('Train size: {}, Test size: {}' .format(X_train.shape, X_test.shape))


# The final thing before learning is to convert these list of words , that came from title and description, into features. I used one hot encoding here but there would be over 15000 features if I had used all words so we count number of occurance for each word and keep only 2500 words which were more frequent. Also because the title did not contain much data to be respected as an independent feature we aggregated it with the description and made one text for processing.

# In[217]:


descCounts = {}
descDictionary = {}
for index, row in X_train.iterrows():
    for word in row['Words']:
        if word not in descCounts:
            descCounts[word] = 1
        else:
            descCounts[word] += 1 
descDictionary = {k: v for k, v in sorted(descCounts.items(), key = lambda item: item[1], reverse = True)}


# In[218]:


print(len(descDictionary))
print(descDictionary)


# In[219]:


descDictionary = dict(itertools.islice(descDictionary.items(), 2500))


# Now, we applied a bag of words method in order to extract trainable features from texts. In this method, we had a column for each frequent word and for each row put 1 in the columns that their word was available in that row's description.

# In[220]:


for key in descDictionary:
    X_train['desc_' + key] = 0
    X_train['desc_' + key] = X_train['Words'].apply(lambda x : 1 if key in x else 0)
    X_test['desc_' + key] = 0
    X_test['desc_' + key] = X_test['Words'].apply(lambda x : 1 if key in x else 0)
X_train = X_train.drop(columns = ['Words'])
X_test = X_test.drop(columns = ['Words'])


# In[221]:


X_train.head(10)


# In[222]:


X_test.head(10)


# # **Logistic Regression Classifier**
# In this section we try to train a model on the text features extracted and skip using images. The first algorithm used is logistig regression which reaches to 48% accuracy. Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable.

# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


# In[48]:


lr = LogisticRegression()
clf = OneVsRestClassifier(lr)
X_train2 = X_train
y_train2 = y_train
X_test2 = X_test
y_test2 = y_test


# In[49]:


X_train2 = X_train2.drop(columns = ['image'])
X_test2 = X_test2.drop(columns = ['image'])


# In[50]:


clf.fit(X_train2, y_train2)
y_pred = clf.predict(X_test2)


# In[51]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test2, y_pred)


# # **Random Forest Classifier**
# 
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. We reache near 59% accuracy with random forest.

# In[52]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
rfc_mdl = RFC(n_estimators=50, max_depth=25, class_weight ='balanced', n_jobs=-1).fit(X_train2, y_train2)
rf_pred = rfc_mdl.predict(X_test2)


# In[ ]:


from sklearn.metrics import hamming_loss
hl = hamming_loss(y_test2, rf_pred)
accuracy_score(y_test2, rf_pred)


# # **ADABoost Classifier**

# An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. We changed the parameters of AdaBoostClassifier function such as max_depth of our tree , n_estimators and learning_rate, and the best accuracy emerges when max_depth=12, n_estimators=50, algorithm="SAMME", learning_rate=0.5.
# 
# SMME algorithm is slower than SAMME.R algorithm(which is the default one), but it brings about higher accuracy in the final result.

# In[250]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


ada_clf = OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=9), n_estimators=50, algorithm="SAMME", learning_rate=0.5))
ada_clf.fit(X_train2, y_train2)
ada_clf_pred = ada_clf.predict(X_test2)
accuracy_score(y_test2, ada_clf_pred)


# In[ ]:


ada_clf = OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=9), n_estimators=100, algorithm="SAMME", learning_rate=0.5))
ada_clf.fit(X_train2, y_train2)
ada_clf_pred = ada_clf.predict(X_test2)
accuracy_score(y_test2, ada_clf_pred)


# In[ ]:


ada_clf = OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=9), n_estimators=50, algorithm="SAMME", learning_rate=0.8))
ada_clf.fit(X_train2, y_train2)
ada_clf_pred = ada_clf.predict(X_test2)
accuracy_score(y_test2, ada_clf_pred)


# In[258]:


ada_clf2 = OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=12), n_estimators=50, algorithm="SAMME", learning_rate=0.5))
ada_clf2.fit(X_train2, y_train2)
ada_clf_pred2 = ada_clf2.predict(X_test2) 
accuracy_score(y_test2, ada_clf_pred2)


# In[ ]:


ada_clf = OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=16), n_estimators=50, algorithm="SAMME", learning_rate=0.5))
ada_clf.fit(X_train2, y_train2)
ada_clf_pred = ada_clf.predict(X_test2) 
accuracy_score(y_test2, ada_clf_pred)


# We reached a high accuracy of hardly 61% which is a realy high number for this calssification problem for 2 reasons:
# - The number of rows with missing values were a lot, about 2500 rows, so the best accuracy theoretically possible is about 75% so our accuracy is actually 81% in the data which were possible to train. Also, the baseline of this problem is about 21% accuracy which is the situation if we classify every film as action which is the most frequent genre.
# - The second problem is the problems of hazm library. It is widely accepted that languauge processing tools available for persian language are less enhanced and developed ones versus the tools available for english language.

# In[260]:


from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
print(classification_report(y_test2, ada_clf_pred2))
print(balanced_accuracy_score(y_test2, ada_clf_pred2))


# In[ ]:


dataframe2.head()      


# Now that we reached an acceptable accuracy for text-based model we divert our attention to the image database available which is the collection of posters of each movie. First of all, we mount the drive and import te images using a shortcut of the given ling in our drive. Then we unzip this rar file containing images.

# In[386]:


import zipfile
from google.colab import drive
drive.mount('/content/drive/')
zip_ref = zipfile.ZipFile("/content/drive/My Drive/train_images.zip", 'r')
zip_ref.extractall("/tmp")
zip_ref.close()


# (This part is here too handle the problems regarding opencv versions)

# In[48]:


# !pip3 install opencv-python==3.4.2.17 opencv-contrib-python==3.4.2.17

# !pip3 uninstall opencv-contrib-python
# !pip3 uninstall opencv-python
# !pip3 install opencv-contrib-python
# !pip3 install opencv-python
# !pip uninstall imgaug && pip uninstall albumentations && pip install git+https://github.com/aleju/imgaug.git

# !pip3 uninstall opencv-contrib-python
# !pip3 uninstall opencv-python
# !pip3 install opencv-contrib-python
# !pip3 install opencv-python

# !pip install opencv-python==3.4.2.16
#! pip install opencv-contrib-python==3.4.2.16


# In[49]:


import cv2 as cv
import numpy as np


# We use 2 methods for image processing of posters available: SIFT and ORB
# # ORB:
# Oriented FAST and Rotated BRIEF (ORB) was developed at OpenCV labs in 2011, as an efficient and viable alternative to SIFT and SURF. ORB was conceived mainly because SIFT and SURF are patented algorithms. ORB, however, is free to use. ORB performs as well as SIFT on the task of feature detection (and is better than SURF) while being almost two orders of magnitude faster. ORB builds on the well-known FAST keypoint detector and the BRIEF descriptor. Both of these techniques are attractive because of their good performance and low cost. Orb algorithm uses a multiscale image pyramid. An image pyramid is a multiscale representation of a single image, that consist of sequences of images all of which are versions of the image at different resolutions. Each level in the pyramid contains the downsampled version of the image than the previous level. Once orb has created a pyramid it uses the fast algorithm to detect keypoints in the image. By detecting keypoints at each level orb is effectively locating key points at a different scale. In this way, ORB is partial scale invariant. After locating keypoints orb now assign an orientation to each keypoint like left or right facing depending on how the levels of intensity change around that keypoint.
# 
# # Sift:
# Scale-Invariant Feature Transform(SIFT) was first presented in 2004. SIFT is invariance to image scale and rotation. This algorithm is patented, so this algorithm is included in the Non-free module in OpenCV(But we used it using a trick in stackoverflow :D ). Real world objects are meaningful only at a certain scale. This multi-scale nature of objects is quite common in nature and a scale space attempts to replicate this concept on digital images. Scale-space is separated into octaves and the number of octaves and scale depends on the size of the original image. So we generate several octaves of the original image. Each octave’s image size is half the previous one. Within an octave, images are progressively blurred using the Gaussian Blur operator. Now we use those blurred images to generate another set of images, the Difference of Gaussians(DoG) which are great for finding out interesting keypoints in the image. The difference of Gaussian is obtained as the difference of Gaussian blurring of an image with two different σ, let it be σ and kσ. This process is done for different octaves of the image in the Gaussian Pyramid. One pixel in an image is compared with its 8 neighbors as well as 9 pixels in the next scale and 9 pixels in previous scales. This way, a total of 26 checks are made. If it is a local extrema, it is a potential keypoint. It basically means that keypoint is best represented in that scale. The rest is like the ORB method.
# 
# 

# We use a method in which we elicit the keypoints available in each picture then we run a k-means clustering algorithm on the list of all keypoints of all pictures and find a predefined number of centers for clusters whithin images keypoints. Then we label each component of each image with the nearest cluster center and count the number of components in each cluster for every image. Then by using zero-one columns we represent the features extracted. In other words we applied a bag of words method for images in order to use image features next to text features. This method is very accurate and can reach high accuracies about 90% but unfortunately, time complexity of this method is high too. So we  used less than 1/10 of dataset and about 2000 to 3000 clusters and e reached a high accuracy of ner 50 with this low value of data. Surely we would have reached a high accuracy if we had the required resources and the paper we used for this method acknowledges this fact.

# In[50]:


import cv2
X_train2 = X_train[:800]
y_train2 = y_train[:800]
X_test2 = X_test[:200]
y_test2 = y_test[:200]
des_list = []
orb = cv2.ORB_create()
location = "/tmp/train_set/"
for index, row in X_train2.iterrows():
    im = cv2.imread(location + row['image'])
    kp = orb.detect(im, None)
    keypoints, descriptor = orb.compute(im, kp)
    des_list.append((location + row['image'], descriptor))


# In[51]:


descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
descriptors_float = descriptors.astype(float)


# In[52]:


descriptors.shape


# In[53]:


from scipy.cluster.vq import kmeans, vq
k = 3000
voc, variance = kmeans(descriptors_float, k, 1)


# In[54]:


im_features = np.zeros((len(des_list), k), "float32")
for i in range(len(des_list)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1


# In[55]:


for i in range(k):
    X_train2[str(i)] = im_features[:, i]


# We used standard scaler to reduce the effect of big numbers in some columns.

# In[56]:


from sklearn.preprocessing import StandardScaler
X_train2 = X_train2.drop(columns = ['image'])
stdslr = StandardScaler().fit(X_train2)
X_train2 = stdslr.transform(X_train2)


# In[57]:


from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter = 80000)
clf.fit(X_train2, np.array(y_train2))


# In[58]:


des_list_test = []
location = "/tmp/train_set/"
for index, row in X_test2.iterrows():
    im = cv2.imread(location + row['image'])
    kp = orb.detect(im, None)
    keypoints, descriptor = orb.compute(im, kp)
    des_list_test.append((location + row['image'], descriptor))


# In[59]:


from scipy.cluster.vq import vq
test_features = np.zeros((len(des_list_test), k), "float32")
for i in range(len(des_list_test)):
    words, distance = vq(des_list_test[i][1], voc)
    for w in words:
        test_features[i][w] += 1


# In[60]:


for i in range(k):
    X_test2[str(i)] = test_features[:, i]


# In[61]:


X_test2 = X_test2.drop(columns = ['image'])
stdslr = StandardScaler().fit(X_test2)
X_test2 = stdslr.transform(X_test2)


# In[62]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test2, clf.predict(X_test2))
print("Accuracy : %f"%accuracy)


# In[63]:


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
model = OneVsRestClassifier(lr)
model.fit(X_train2, y_train2)
y_pred = model.predict(X_test2)
accuracy_score(y_test2, y_pred)


# In[64]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
rfc_mdl = RFC(n_estimators=50, max_depth=25, class_weight ='balanced', n_jobs=-1).fit(X_train2, y_train2)
rf_pred = rfc_mdl.predict(X_test2)


# In[65]:


from sklearn.metrics import hamming_loss
hl = hamming_loss(y_test2, rf_pred)
accuracy_score(y_test2, rf_pred)


# In[66]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf = OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=9), n_estimators=50, algorithm="SAMME", learning_rate=0.5))
ada_clf.fit(X_train2, y_train2)
ada_clf_pred = ada_clf.predict(X_test2)
accuracy_score(y_test2 , ada_clf_pred)


# Now we do the last equence of actions taken for ORB and use SIFT instead.

# In[67]:


# !pip install opencv-python==3.4.2.16
# !pip install opencv-contrib-python==3.4.2.16


# In[68]:


import cv2 as cv
X_train2 = X_train[:800]
y_train2 = y_train[:800]
X_test2 = X_test[:200]
y_test2 = y_test[:200]
des_list = []
sift = cv.xfeatures2d.SURF_create()
location = "/tmp/train_set/"
for index, row in X_train2.iterrows():
    im = cv.imread(location + row['image'])
    keypoints, descriptor = sift.detectAndCompute(im, None)
    des_list.append((location + row['image'], descriptor))


# In[69]:


descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
descriptors_float = descriptors.astype(float)
descriptors.shape


# In[70]:


from scipy.cluster.vq import kmeans, vq
k = 1000
voc, variance = kmeans(descriptors_float, k, 1)


# In[72]:


im_features = np.zeros((len(des_list), k), "float32")
for i in range(len(des_list)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1
for i in range(k):
    X_train2[str(i)] = im_features[:, i]
from sklearn.preprocessing import StandardScaler
X_train2 = X_train2.drop(columns = ['image'])
stdslr = StandardScaler().fit(X_train2)
X_train2 = stdslr.transform(X_train2)


# In[73]:


from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter = 80000)
clf.fit(X_train2, np.array(y_train2))
des_list_test = []
location = "/tmp/train_set/"


# In[74]:


for index, row in X_test2.iterrows():
    im = cv.imread(location + row['image'])
    keypoints, descriptor = sift.detectAndCompute(im, None)
    des_list_test.append((location + row['image'], descriptor))
from scipy.cluster.vq import vq
test_features = np.zeros((len(des_list_test), k), "float32")
for i in range(len(des_list_test)):
    words, distance = vq(des_list_test[i][1], voc)
    for w in words:
        test_features[i][w] += 1
for i in range(k):
    X_test2[str(i)] = test_features[:, i]


# In[75]:


X_test2 = X_test2.drop(columns = ['image'])
stdslr = StandardScaler().fit(X_test2)
X_test2 = stdslr.transform(X_test2)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test2, clf.predict(X_test2))
print("Accuracy : %f"%(accuracy * 100))


# In[76]:


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
model = OneVsRestClassifier(lr)
model.fit(X_train2, y_train2)
y_pred = model.predict(X_test2)
accuracy_score(y_test2, y_pred)


# In[77]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
rfc_mdl = RFC(n_estimators=50, max_depth=25, class_weight ='balanced', n_jobs=-1).fit(X_train2, y_train2)
rf_pred = rfc_mdl.predict(X_test2)
from sklearn.metrics import hamming_loss
hl = hamming_loss(y_test2, rf_pred)
accuracy_score(y_test2, rf_pred)


# In[78]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf = OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=9), n_estimators=50, algorithm="SAMME", learning_rate=0.5))
ada_clf.fit(X_train2, y_train2)
ada_clf_pred = ada_clf.predict(X_test2)
accuracy_score(y_test2 , ada_clf_pred)


# The conclusion of this part is taht we reached about 45% accuracy in the best case with only 1/10 of dataset. We could have reached a higher accuracy if problems below get fixed.
# - If we had better system with more computtional ability we would have used all of dataset and would have reached more than 80% accuracy like paper suggests.
# - If we had more memory available we could have use more keypoints instead of 3000 because higher numbers faced ram limitations and it is obvious that by more keypoints stored, more accuracy will be reachable.
# - Last of all, if more rows had description cell, the text processing section would have reached a higher accuracy.

# In the last part, we used a known technic called transfer learning in which we extract some features using pixels of images and a convoloutional neural network and pass it to another model to use all possible data given to us. The higher model can be another CNN too. Transfer learning generally refers to a process where a model trained on one problem is used in some way on a second related problem. In deep learning, transfer learning is a technique whereby a neural network model is first trained on a problem similar to the problem that is being solved. One or more layers from the trained model are then used in a new model trained on the problem of interest. Transfer learning has the benefit of decreasing the training time for a neural network model and can result in lower generalization error.

# In[383]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import cv2 as cv
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Movie posters are a key component in the film industry. It is the primary design element that captures the viewer's attention and conveys a movie's theme. Human emotions are aroused by colour, brightness, saturation, hues, contours etc in images. Therefore, we are able to quickly draw general conclusions about a movie's genre (comedy, action, drama, animation etc) based on the colours, facial expressions and scenes portrayed on a movie poster. This leads to the assumption that the colour information, texture features and structural cues contained in images of posters, possess some inherent relationships that could be exploited by ML algorithms for automated prediction of movie genres from posters.
# 
# In this project, CNNs are trained on movie poster images to identify the genres of a movie from its poster. This is a multi-label classification task, since a movie can have multiple genres linked to it, i.e have an independent probability to belong to each label (genre).
# 
# The implementation is based on Keras and Tensorflow.

# In[381]:


def resize_img(path):
  try:
    img = cv.imread(path)
    img = cv.resize(img, (75, 115))
    img = img.astype(np.float32)/255
    return img
  except Exception as e:
    print(str(e))
    return None


# We will first create an instance of ImageDataGenerator for both training and validation purposes. As pixel values range from 0 to 255 we will normalize them in range 0 to 1. To do this we will pass in the argument (rescale = 1./255) when creating an instance of ImageDataGenerator. After this, we will use the .flow_from_directory() method of the instance to label the images for both directories and store the result in train_generator and validation_generator for training and validation purposes. While calling this method we will pass in the target_size attribute to ensure that our images in the dataset are of the same size.

# In[388]:


val_imgs = []
location = "/tmp/train_set/"
i = 0
for index, row in dataframe.iterrows():
  if i % 1000 == 0:
    print("Processing i:", i, str(row['image']))
  img = resize_img(location + str(row['image']))
  if img is not None:
    val_imgs.append(img)
  i += 1


# In[391]:


plt.imshow(val_imgs[11])


# In[390]:


val_np_imgs = np.array(val_imgs)
val_np_imgs.shape


# In[392]:


x_train_2 ,x_test_2, y_train_2, y_test_2 = train_test_split(val_np_imgs, y, random_state = 7, test_size = 0.2)


# In[393]:


y_train_2.values.reshape((-1,1))


# Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation function and a weight constraint of max norm set to 3.
# Dropout set to 20%.
# Convolutional layer, 32 feature maps with a size of 3×3, a rectifier activation function and a weight constraint of max norm set to 3.
# Max Pool layer with size 2×2.
# Flatten layer.
# Fully connected layer with 128 units and a rectifier activation function.
# Dropout set to 50%.
# Fully connected output layer with 10 units and a softmax activation function.
# A logarithmic loss function is used with the stochastic gradient descent optimization algorithm configured with a large momentum and weight decay start with a learning rate of 0.01.

# In[394]:


from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv2D,MaxPool2D

# Initialising the CNN
model = Sequential()

#First Convulation Layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape = x_train_2[0].shape ))
model.add(BatchNormalization())
#Pooling
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

#Second Convulation Layer
model.add(Conv2D(64,(5,5),activation='relu',input_shape = x_train_2[0].shape ))
model.add(BatchNormalization())
#Pooling
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

#Third Convulation Layer
model.add(Conv2D(128,(3,3),activation='relu',input_shape = x_train_2[0].shape ))
model.add(BatchNormalization())
#Pooling
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

#Fourth Convulation Layer
model.add(Conv2D(128,(3,3),activation='relu',input_shape = x_train_2[0].shape ))
model.add(BatchNormalization())
#Pooling
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

#Flattening Layer
model.add(Flatten()) 

#First full connection Layer
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Second full connection Layer
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


#Output Layer
model.add(Dense(10, activation='sigmoid'))


# In[395]:


model.summary()


# In[396]:


model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[379]:


history = model.fit(x_train_2, y_train_2, epochs = 10, validation_data = (x_test_2, y_test_2))


# In[400]:


print("Accuracy :", history.history['val_acc'][-1])


# The final step is to predict for the evaluation dataset given dataset. First we uploaded the csv file.

# In[223]:


from google.colab import files
uploaded = files.upload()


# In[224]:


import io
data2 = pd.read_csv(io.BytesIO(uploaded['eval_set.csv']))
data2.head(10)


# In[342]:


data = data2


# Now we apply the preprocesses we did in the training phase. First we droped video_id then replaced all cells with only space instead of value with NaN.

# In[343]:


data = data.drop(columns = ['video_id'])


# In[344]:


data = data.replace(r'^\s*$', np.NaN, regex=True)


# In[345]:


data.info()


# Now, we filled the NaN values like we did in training phase.

# In[346]:


data['age_rating'].fillna(dataframe['age_rating'].mean(), inplace = True)


# In[347]:


data['country_fa'].fillna("آمریکا", inplace = True)


# In[348]:


data['age_rating'] = dataframe['age_rating'].astype(int)


# In[349]:


data['duration'] = pd.to_datetime(data['duration'])


# In[350]:


data['hour'] = data['duration'].dt.hour
data['minute'] = data['duration'].dt.minute


# In[351]:


data['minute'] = data['minute'].replace(0, np.NaN)


# In[352]:


data = data.apply(dateHandler, axis = 'columns')
data = data.drop(columns = ['hour', 'duration'])


# In[353]:


data['minute'] = data['minute'].fillna(data['minute'].mean())
data['age_rating'].fillna(dataframe['age_rating'].mean(), inplace = True)
data.isnull().sum()


# In[354]:


data.head(10)


# In[355]:


data['production_year'] = data['production_year'].astype(float)
data['production_year'] = data['production_year'].astype(int)
data['minute'] = data['minute'].astype(int)


# In[356]:


data = data.drop(columns = ['production_year'])


# In[357]:


data.isnull().sum()


# In[358]:


for column in X_train.columns:
    if column not in data.columns:
        data[column] = 0


# In[359]:


data = data.drop(columns = ['country_fa'])


# In[360]:


data['description_fa'] = data['description_fa'].astype(str)
data['title_fa'] = data['title_fa'].astype(str)


# The next step is to do the processes needed for text processing.

# In[361]:


def preprocessTextofColumn(columnName):
    normalizer = hazm.Normalizer()
    tokenizer = hazm.WordTokenizer()
    lemmatizer = hazm.Lemmatizer()
    ENGStopWords = set(stopwords.words('english'))
    PERStopWords = set(hazm.utils.stopwords_list())
    PERStopWords = PERStopWords.union({':','ی','ای',',','،','(',')',':',';','-','_','.','/','+','=','?'})
    stopWords = ENGStopWords.union(PERStopWords)
    allWords = []
    for index, row in data.iterrows():
        text = row['title_fa'] + ' ' + row[columnName]
        normalizedText = normalizer.affix_spacing(text)
        words = tokenizer.tokenize(normalizedText)
        filteredWords = []
        for word in words:
            if not word in stopWords:
                filteredWords.append(lemmatizer.lemmatize(word.lower()))
        allWords.append(filteredWords)
    newColumnName = 'Words'
    data[newColumnName] = allWords
    return data


# In[362]:


data = preprocessTextofColumn('description_fa')


# In[363]:


data = data.drop(columns = ['description_fa', 'title_fa'])


# In[364]:


for key in descDictionary:
    data['desc_' + key] = 0
    data['desc_' + key] = data['Words'].apply(lambda x : 1 if key in x else 0)
data = data.drop(columns = ['Words'])


# In[365]:


data = data.drop(columns = ['image'])


# In[366]:


data.head(10)


# Now we predict using the best model we developed during this project which was adaboost and saved our predictions in a csv file named predictions.csv and upoaded it with other files.

# In[367]:


ada_clf_pred3 = ada_clf2.predict(data)


# In[370]:


ada_clf_pred3


# In[406]:


import csv
with open("predictions.csv", 'w', newline='\n') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(ada_clf_pred3)


# # Conclusion:
# In this project, we reached an accuracy of 61% using adaboost and predicted the calsses for the given validation dataset. As we mentioned before, we would have reached a better accuracy if below problems get fixed:
# - The number of rows with missing values were high, about 2500 rows, so the best accuracy theoretically possible is about 75%. Also, the baseline of this problem is about 21% accuracy which is the situation if we classify every film as action which is the most frequent genre.
# - The second problem is the problems of hazm library. It is widely accepted that languauge processing tools available for persian language are less enhanced and developed ones versus the tools available for english language.
# - The third problem is the computational power and storag available to us.
# - Unfortunately, we did not have enough resources and gpu to complete transfer learning method discussed before and only used CNN for images and did not use all information we had.
