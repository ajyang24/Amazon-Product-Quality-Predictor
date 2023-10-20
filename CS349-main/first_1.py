from functools import total_ordering
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


product_training = pd.read_json("devided_dataset_v2/Grocery_and_Gourmet_Food/train/product_training.json")
review_training = pd.read_json("devided_dataset_v2/Grocery_and_Gourmet_Food/train/review_training.json")

# Calculate the features, such as sentiment, verified purshase, votes, etc
def compute_features():
    # A dictionary to hold asins and their corresponding compounds and number of reviews
    products = {}

    # Dictionary
    # Key = Asin
    # Value = [compound review, # reviews, bought, not bought,votes, compound summary, # summaries, unix]
    # compound review = 0
    # reviews = 1
    # bought = 2
    # not bought = 3
    # votes = 7
    # compound summary = 5
    # summaries = 6
    #  unix = 4

    sia = SentimentIntensityAnalyzer()

    def compound_score(sentence):
      sentiment_dict = sia.polarity_scores(sentence)
      return sentiment_dict['compound']

    # .shape (rows, col) 0 will be rows
    
    # Loop over every row
    for i in range(review_training.shape[0]):
        # Variables for metrics  of the row
        current_review = review_training.loc[i]
        current_compound = 0
        current_bought = 0
        current_not_bought = 0
        current_votes = 0
        current_unix = 0
        current_image = 0
        current_summary = 0
        # Feature 1: Compute Senitment
        if current_review['reviewText'] != None:
            current_compound = compound_score(current_review['reviewText'])
        # Feature 2: Compute bought or not 
        if current_review['verified']:
            current_bought = 1
        else:
            current_not_bought = 1
        # Number of votes: Feature 3
        # if current_review['vote']:
        #     current_votes = current_review['vote']
        # Feature 4: Unix Time
        if current_review['unixReviewTime'] != None:
            current_unix = current_review['unixReviewTime'] 
        # Feature 5: Image added
        if current_review['image'] != None:
            current_image = 1
        # Feature 6: Summary Sentiment
        if current_review['summary'] != None:
            current_summary = compound_score(current_review['summary'])
        # Add data to dictionary
        if current_review['asin'] in products:
            # 1: Compund
            products[current_review['asin']][0] += current_compound
            # 2: Number of reviews
            products[current_review['asin']][1] += 1
            # 3: bought
            products[current_review['asin']][2] += current_bought
            # 4: not bought
            products[current_review['asin']][3] += current_not_bought
            # 5: total unix time
            products[current_review['asin']][4] += current_unix
            # 6: Image added
            products[current_review['asin']][5] += current_image
            # 7: Summary
            products[current_review['asin']][6] += current_summary
        else:
            products[current_review['asin']] = [current_compound, 1, current_bought, current_not_bought, current_unix, current_image, current_summary]

    return products

# Stores products into a pickle file for easier training process
def store_data(products):
    # Store Product dict to speed up sentiment analysis for future runs
    with open('sentiment.pkl', 'wb') as fp:
        pickle.dump(products, fp)
        print('dictionary saved successfully to file')

# Uses pickle file to recall features stored
def recall_data():
    # Recall stores sentiment analysis from file
    read_product = {}
    with open('sentiment.pkl', 'rb') as fp:
        read_product = pickle.load(fp)
        # print(read_product)
    return read_product

# Compute final features and run the model on features
def use_model(read_product):

    # Inner function to compute final version of features
    def final_features(read_product):
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
        return read_product

    # Inner function to adjust size and shape of features
    def reshape_features(read_product, target_df):
        # Change dict to df
        data_list = [(key, *values) for key, values in read_product.items()]
        # Convert the list of tuples to a DataFrame
        feature_df = pd.DataFrame(data_list, columns=['asin', '0', '1', '2', '3', '4', '5', '6'])
        # Merge data frames
        final_df = pd.merge(feature_df, target_df, on='asin')

        return final_df

    # Inner function to run model on final features
    def run_model(final_df):
        # Select features to train
        shaped_df_features = final_df[['0']]
        shaped_df_targets = final_df[['awesomeness']]

        # print(shaped_df_features)
        # print(shaped_df_targets)

        # Train Test Split using k-fold verifcation
        k = 10
        totalP = 0
        totalR = 0
        totalF = 0
        for i in range(k):
            X_train, X_test, y_train, y_test = train_test_split(shaped_df_features, shaped_df_targets, test_size=0.2, random_state=i)
            
            # Choose a classifier
            # clf = SVC()
            clf = GaussianNB()
            #clf = LogisticRegression()

            # Train the classifier on the training data
            clf.fit(X_train, y_train.values.ravel())

            # Test the classifier on the testing data
            y_pred = clf.predict(X_test)

            # Evaluate the performance of the classifier
            precisionRecallFscore = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            totalP += precisionRecallFscore[0]
            totalR += precisionRecallFscore[1]
            totalF += precisionRecallFscore[2]

        
        print(totalP/k)
        print(totalR/k)
        print(totalF/k)
    target_df = product_training[['asin','awesomeness']]

    # Function calls
    read_product = final_features(read_product)
    # getting awesomeness
    final_df = reshape_features(read_product, target_df)
    run_model(final_df)

def main():
    # Read files
    product_training = pd.read_json("devided_dataset_v2/Grocery_and_Gourmet_Food/train/product_training.json")
    review_training = pd.read_json("devided_dataset_v2/Grocery_and_Gourmet_Food/train/review_training.json")

    # products = {}
    # products = compute_features()
    # store_data(products)
    products = recall_data()
    print(products)
    use_model(products)



if __name__ == "__main__":
    main()



# Just for reference on what a row looks like

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