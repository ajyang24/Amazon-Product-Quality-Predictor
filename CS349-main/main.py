import pandas as pd
import pickle
import numpy as np
import warnings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.exceptions import ConvergenceWarning
# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Read files
product_training = pd.read_json("./Grocery_and_Gourmet_Food/train/product_training.json")
review_training = pd.read_json("./Grocery_and_Gourmet_Food/train/review_training.json")

product_training1 = pd.read_json("./Grocery_and_Gourmet_Food/test1/product_test.json")
review_training1 = pd.read_json("./Grocery_and_Gourmet_Food/test1/review_test.json")

# Calculate the features, such as sentiment, verified purshase, votes, etc
def compute_features(review_training):
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
def use_model(read_product, read_product2):

    # Inner function to compute final version of features
    def final_features(read_product):
        # Average compound: Store at index 0
        # Ratio of votes: Store at index 2
        # Average Unix Time: 
        for entry in read_product:
            read_product[entry][0] /= read_product[entry][1]
            # Plus 1 to avoid dive by 0 errors
            read_product[entry][2] += 1
            read_product[entry][2] /= (read_product[entry][3] + 1)
            # Average Unix
            read_product[entry][4] /= read_product[entry][1]
            # Average Image
            read_product[entry][5] /= read_product[entry][1]
            # Average Summary Sentiment
            read_product[entry][6] /= read_product[entry][1]

        return read_product

    # Inner function to adjust size and shape of features
    def reshape_features(read_product):
        # Change dict to df
        data_list = [(key, *values) for key, values in read_product.items()]
        # Convert the list of tuples to a DataFrame
        feature_df = pd.DataFrame(data_list, columns=['asin', '0', '1', '2', '3', '4', '5', '6'])
        # Merge data frame
        return feature_df

    def merge_features(feature_df, target_df):
        return pd.merge(feature_df, target_df, on='asin')

    # Inner function to run model on final features
    def run_model(final_df):
        # Select features to train
        shaped_df_features = final_df[['0','2']]
        shaped_df_targets = final_df[['awesomeness']]

        # print(shaped_df_features)
        # print(shaped_df_targets)

        # Train Test Split using k-fold verifcation (removed for final predictions file)

        # Keep track of precision, recall, and f1 scores
        totalP = 0
        totalR = 0
        totalF = 0
        X_train, X_test, y_train, y_test = train_test_split(shaped_df_features, shaped_df_targets, test_size=2, random_state=42)
        
        def single_pred():
            # Choose a classifier
            # clf = SVC()
            # clf = GaussianNB()

            # {'solver': 'newton-cg', 'penalty': 'none', 'max_iter': 1000, 'l1_ratio': 0.5, 'fit_intercept': False, 'C': 0.0018329807108324356}
            # {'solver': 'newton-cg', 'penalty': 'none', 'max_iter': 1000, 'l1_ratio': 0.5, 'fit_intercept': False, 'C': 0.0018329807108324356}
            # {'solver': 'saga', 'penalty': 'elasticnet', 'max_iter': 1000, 'l1_ratio': 0.7000000000000001, 'fit_intercept': False, 'C': 0.23357214690901212}
            # clf = LogisticRegression(solver='newton-cg',penalty=None, max_iter=1000, fit_intercept=False)
            clf = LogisticRegression(solver='sag',penalty=None, max_iter=1000, fit_intercept=False)

            # clf = KNeighborsClassifier(weights='uniform', n_neighbors=43, metric='minkowski')
            # clf = DecisionTreeClassifier(min_samples_split=21, min_samples_leaf=1, max_features='log2', max_depth=10, criterion='entropy')
            # clf = RandomForestClassifier(n_estimators=100, random_state=42)

            # Train the classifier on the training data
            clf.fit(X_train, y_train.values.ravel())

            # # Test the classifier on the testing data
            # y_pred = clf.predict(X_test)

            return clf
        
        def hyper_optimization(): 
            # # Hyper Paramter Optimization: KNN
            # clf = KNeighborsClassifier()
            # param_grid = {
            #     'n_neighbors': np.arange(1, 50),
            #     'weights': ['uniform', 'distance'],
            #     'metric': ['euclidean', 'manhattan', 'minkowski']
            # }
            # # grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)

            # # Hyper Parameter Optimization: SVC
            # clf = SVC()
            # # Define the hyperparameter search space
            # param_grid = {
            #     'C': np.logspace(-3, 3, 7),
            #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            #     'degree': [2, 3, 4, 5],
            #     'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),
            #     'coef0': np.linspace(-1, 1, 21),
            # }
            
            # # Hyper Parameter Optimization: Decision Tree
            # clf = DecisionTreeClassifier()

            # # Define the hyperparameter search space
            # param_grid = {
            #     'criterion': ['gini', 'entropy'],
            #     'max_depth': [None] + list(range(10, 100, 10)),
            #     'min_samples_split': range(2, 100, 10),
            #     'min_samples_leaf': range(1, 100, 10),
            #     'max_features': [None, 'sqrt', 'log2'] + list(np.arange(0.1, 1.1, 0.1)),
            # }

            # Hyper Parameter Optimization:Logistic Regression
            clf = LogisticRegression()

            param_grid = {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': np.logspace(-4, 4, 20),
                'fit_intercept': [True, False],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 300, 500, 1000],
                'l1_ratio': np.linspace(0, 1, 21)  # Only used if penalty='elasticnet'
            }

            # # Hyper Parameter Optimization: Gaussian Naive Bayes
            # clf = GaussianNB()

            # # Define the hyperparameter search space
            # param_grid = {
            #     'var_smoothing': np.logspace(-10, -8, 50)
            # }

            # grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
            grid_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=50, cv=5, scoring='f1', n_jobs=-1, random_state=42)

            grid_search.fit(X_train, y_train.values.ravel())
            print("Best hyperparameters:", grid_search.best_params_)
            y_pred = grid_search.predict(X_test)

            return y_pred

        
        clf = single_pred()

        return clf
        # y_pred = hyper_optimization()
        # Evaluate the performance of the classifier
        # precisionRecallFscore = precision_recall_fscore_support(y_test, y_pred, average='micro')
        # totalP += precision_score(y_test, y_pred)
        # totalR += recall_score(y_test, y_pred)
        # totalF += f1_score(y_test, y_pred)

        # # Average metrics
        # totalR /= k
        # totalP /= k
        # totalF /= k
        
        # print(f'Precision: {totalP}, Recall: {totalR}, F1: {totalF}')

    
    # Calling functions and running code
    target_df = product_training[['asin','awesomeness']]
    # Function calls
    read_product = final_features(read_product)
    read_product2 = final_features(read_product2)
    # reshape feautures
    feature_df = reshape_features(read_product)
    feature_df2 = reshape_features(read_product2)
    # Merge features
    final_df = merge_features(feature_df, target_df)
    # getting awesomeness
    clf = run_model(final_df)
    # Turning predictions into df
    predictions = clf.predict(feature_df2[['0','2']])
    feature_df2['awesomeness'] = pd.Series(predictions.ravel(), name='temp')
    to_file_df = feature_df2[['asin', 'awesomeness']]

    # File creation
    json_str = to_file_df.to_json()

    with open('predictions.json', 'w') as f:
        f.write(json_str)

def main():

    # Training Data
    products = {}
    products = compute_features(review_training)

    # Testing Data
    products2 = {}
    products2 = compute_features(review_training1)

    # # There to help speed up sentiment computation
    # store_data(products)
    # products = recall_data()
    
    use_model(products, products2)



if __name__ == "__main__":
    main()



# Just for reference on what a review looks like

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