import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 


product_training = pd.read_json("./train/product_training.json")
review_training = pd.read_json("./train/review_training.json")


# Recall stores sentiment analysis from file
read_product = {}
with open('sentiment.pkl', 'rb') as fp:
    read_product = pickle.load(fp)
    # print(read_product)

# Average compound: Store at index 0
# Ratio of votes: Store at index 2
for entry in read_product:
    read_product[entry][0] /= read_product[entry][1]
    read_product[entry][2] += 1
    read_product[entry][2] /= (read_product[entry][3] + 1)
    # if read_product[entry][0] > 0.5:
    #     read_product[entry][1] = 1
    # else:
    #     read_product[entry][1] = 0

# print(read_product)
# Change dict to df

data_list = [(key, *values) for key, values in read_product.items()]

# Convert the list of tuples to a DataFrame
feature_df = pd.DataFrame(data_list, columns=['asin', '0', '1', '2', '3'])


# getting awesomeness
target_df = product_training[['asin','awesomeness']]

# Merge data frames
final_df = pd.merge(feature_df, target_df, on='asin')
# print(feature_df)
# print(target_df)
# print(final_df)

# Add more features in drop for testing
shaped_df_features = final_df[['0']]
shaped_df_targets = final_df[['awesomeness']]

print(shaped_df_features)
print(shaped_df_targets)
# Train Test Split using k-fold verifcation
k = 10

for i in range(k):
    X_train, X_test, y_train, y_test = train_test_split(shaped_df_features, shaped_df_targets, test_size=0.2, random_state=i)
    # Choose a classifier (Logistic Regression)
    # clf = SVC()
    clf = GaussianNB()
    # clf = LinearRegression()

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Test the classifier on the testing data
    y_pred = clf.predict(X_test)

    # Evaluate the performance of the classifier
    precisionRecallFscore = precision_recall_fscore_support(y_test, y_pred, average='micro')
    print(precisionRecallFscore)


# asin                               71F1F9B34E46A7F11E8B72C26CA861B6
# reviewerID                         D0A66D9D59DFB8808DE637FA50025550
# unixReviewTime                                           1438560000
# vote                                                           None
# verified                                                       True
# reviewTime                                               08 3, 2015
# style             {'Size:': ' 3.5-oz', 'Style:': ' Spinner Gift ...
# reviewerName                       13BDF4D2B1DF0B4CB2CADE6A71D51E79
# reviewText        These jelly beans were disgusting and perfect ...
# summary                                       Disgustingly perfect.
# image                                                          None