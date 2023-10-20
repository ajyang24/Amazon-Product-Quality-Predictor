from distutils.command.clean import clean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
plt.style.use("fivethirtyeight")
import seaborn as sns
sns.set_style("darkgrid")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import re, string, nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers import Embedding
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


product_training = pd.read_json("devided_dataset_v2/Grocery_and_Gourmet_Food/train/product_training.json")
review_training = pd.read_json("devided_dataset_v2/Grocery_and_Gourmet_Food/train/review_training.json")


def clean_all_data(data):
    review_training1 = data[~data['reviewText'].isnull()]

    def reshape_features(review_df, target_df):
            feature_df = pd.DataFrame(review_df, columns=['asin', 'reviewText'])
            # Merge data frames
            final_df = pd.merge(feature_df, target_df, on='asin')
            final_df = pd.DataFrame(final_df, columns=['reviewText', 'awesomeness'])
            return final_df

    review_df = review_training1
    target_df =  product_training[['asin','awesomeness']]
    review_training1 = reshape_features(review_df, target_df)

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    #used later 
    nltk.download('punkt')

    def clean_text(df, field):
        df[field] = df[field].str.replace(r"@"," at ")
        df[field] = df[field].str.replace("#[^a-zA-Z0-9_]+"," ")
        df[field] = df[field].str.replace(r"[^a-zA-Z(),\"'\n_]"," ")
        df[field] = df[field].str.replace(r"http\S+","")
        df[field] = df[field].str.lower()
        print(df)
        return df

    clean_text(review_training1,"reviewText")

    # Applying Lemmmatizer to remove tenses from texts.
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub('[^a-zA-Z0-9]',' ',text)
        #text= re.sub(emoji.get_emoji_regexp(),"",text)
        text = [lemmatizer.lemmatize(word) for word in text.split() if not word in set(stopwords.words('english'))]
        text = ' '.join(text)
        return text


    review_training1["reviewTextClean"] = review_training1["reviewText"].apply(preprocess_text)
    review_training = review_training1[["awesomeness",'reviewTextClean']]
    return review_training


# Stores products into a pickle file for easier training process
def store_data(data):
    # Store Product dict to speed up sentiment analysis for future runs
    with open('sentimentLSTM.pkl', 'wb') as fp:
        pickle.dump(data, fp)
        print('dictionary saved successfully to file')


# Uses pickle file to recall features stored
def recall_data():
    # Recall stores sentiment analysis from file
    read_product = {}
    with open('sentimentLSTM.pkl', 'rb') as fp:
        read_product = pickle.load(fp)
        # print(read_product)
    return read_product


#Deep Learning 
def DL_model(data):
    X = data["reviewTextClean"]
    y = data.awesomeness
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    # using tokenizer to transform text messages into training and testing set
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_seq_padded = pad_sequences(X_train_seq, maxlen=64)
    X_test_seq_padded = pad_sequences(X_test_seq, maxlen=64)

    X_train_seq_padded[0]

    # construct model
    # can change batch size 
    BATCH_SIZE = 100

    from keras.utils.vis_utils import plot_model
    model = Sequential()
    model.add(Embedding(len(tokenizer.index_word)+1,64))
    model.add(Bidirectional(LSTM(100, dropout=0,recurrent_dropout=0)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1,activation="sigmoid"))

    model.compile("adam","binary_crossentropy",metrics=["accuracy"])
    model.summary()

    # Used for preventing ovefitting
    from keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss",patience=5,verbose=True)

    # Change number of epochs 
    history = model.fit(X_train_seq_padded, y_train,batch_size=BATCH_SIZE,epochs=1,
                        validation_data=(X_test_seq_padded, y_test),callbacks=[early_stop])


    def evaluate_roc():
        pred_train = model.predict(X_train_seq_padded)
        pred_test = model.predict(X_test_seq_padded)
        print('LSTM Recurrent Neural Network baseline: ' + str(roc_auc_score(y_train, pred_train)))
        print('LSTM Recurrent Neural Network: ' + str(roc_auc_score(y_test, pred_test)))


    def evaluate_accuracy():
        model.evaluate(X_test_seq_padded, y_test)

    evaluate_roc()
    evaluate_accuracy()



def main():
    # Read files
    product_training = pd.read_json("devided_dataset_v2/Grocery_and_Gourmet_Food/train/product_training.json")
    review_training = pd.read_json("devided_dataset_v2/Grocery_and_Gourmet_Food/train/review_training.json")
    # To test on partial dataset:
    # product_training = product_training[0:5000]
    # review_training = review_training[0:5000]

    review_training = clean_all_data(review_training)
    # Use store_data and recall_data once to store all clean data as file 
    # store_data(review_training)
    # review_training = recall_data()

    DL_model(review_training)



if __name__ == "__main__":
    main()
