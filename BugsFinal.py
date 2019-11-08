# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 02:40:56 2019

@author: Mihail Mihaylov
"""
from __future__ import absolute_import, division, print_function
from datetime import datetime
from time import time
import argparse
import warnings
import matplotlib.pyplot as plt
from os import path

warnings.filterwarnings('ignore')

import tensorflow as tf
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Dense, Conv1D, Embedding, Dropout, AveragePooling1D, MaxPooling1D
from keras.layers import concatenate, BatchNormalization, Flatten, SpatialDropout1D, LSTM
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import Input
from keras.regularizers import l2
from keras import backend as K

CATEGORIES = 52 ## Number of categories for classification
MAX_LENGTH = 512 ## The maximum length of the text features after Tokenization
MAX_TIME = 365 ## The maximum resolution time considered for the model
SEED = 12345 ## Random seed

def main(model_comb=0, regress=False, learning_rate=None, epochs=None, batches=None):
    tf.set_random_seed(SEED) ## Set the random seed for tensorflow
    np.random.seed(SEED) ## Set the random seed for numpy
    priority = False ## Initially set priority to False
    start = time() ## Start time to calculate how long it took to train a model

    ## Check if model has been already trained before
    r_path = "./Models/bugs-{}-{}-{}-{}-{}-.ckpt".format(model_comb, regress, learning_rate, epochs, batches)
    if path.exists(r_path): ## If model exists load from file
        restore = True
    else: ## Else train a new model
        restore = False 
    
    ## Save model and logs to tensorboard
    checkpoint_path = "./Models/bugs-{}-{}-{}-{}-{}-.ckpt".format(model_comb, regress, learning_rate, epochs, batches)
    cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=0, period=1)
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    tensorboard = TensorBoard(log_dir="./logs/run-{}".format(now), write_graph=True, update_freq="batch")
    
    if model_comb > 6: ## Model combinations from 6 to 12 are a classification for the Priority
        global CATEGORIES
        CATEGORIES = 5 ## Change the number of CATEGORIES to 5 because there is only 5 classes for Priority
        priority = True ## Change the priority to true
        regress = False ## Set regress to false
    
    bugs = prep_data(priority) ##Load and clean up the initial data set
    
    bugs = additional_columns(bugs) ## Add sentiment and length

    if model_comb in {1, 4, 7, 10}: ## Those 4 models require OneHotEncoding for version and component
        bugs = onehot_prep(bugs) ## One-hot encode the Version and Component columns
            
    bugs = text_prep(bugs) ## Prepare the text features
    bugs = labels_prep(bugs, regress) ## Prepare the lables 
    
    ## Split the dataset into Training and Testing sets and further into Numeric features, Text features and lables
    X_train_text, X_train_feat, X_test_text, X_test_feat, y_train, y_test = split_data(bugs, priority, regress)
    
    ## Tokenize the text features
    X_train_text, X_test_text, vocab_size = tokenize(X_train_text, X_test_text)
    
    ## Train and evaluate a baseline model
    model = baseline(X_train_text, y_train, X_test_text, y_test, regress)
    
    print("Building model {}...".format(model_comb))
    if model_comb in {1, 2, 4, 5, 7, 8, 10, 11}:  ## Use network one in these 8 cases since they involve both text and numeric features
        feat = X_train_feat.shape[1]
        model = network_one(feat, model_comb, vocab_size, learning_rate, epochs, batches, regress)    
        X = [X_train_feat, X_train_text]
    elif model_comb in {3, 6, 9, 12}: ## In the remaining four cases use network two which only has text features 
        model = network_two(model_comb, vocab_size, learning_rate, epochs, batches, regress)
        X = X_train_text
    
    if not restore: ##If restore is False train a new model
        print("Training model {}...".format(model_comb))
        history = model.fit(X, y_train, batch_size=batches, epochs=epochs, verbose=1,
                  validation_split=0.1, shuffle='batch', callbacks = [cp_callback, tensorboard])        
        plot_model_training(history, regress)
        ## Calculate the time it took to train the model
        end = time()
        print("The model took {} minutes and {} seconds to train.".format(np.int32((end-start)//60), np.int32((end-start)%60)))
    else: ## If restore is True load the weights from the checkpoint file
        model.load_weights(r_path)
        print("Model Restored!")

        print("Testing the model on the Training set...")
        test_model(X_train_text, X_train_feat, y_train, model, regress, model_comb)
    
    
    plot_model(model, to_file='./Graphs/model{}{}.png'.format(model_comb, regress))
    
    ## Test the trained model on the testing set
    print("Testing the model on the Test set...")
    test_model(X_test_text, X_test_feat, y_test, model, regress, model_comb)
    
    return bugs, X_train_text, X_train_feat, X_test_text, X_test_feat, y_train, y_test, model

## Load the dataset and do some initial cleaning on it
def prep_data(priority):
    print("Preparing initial data...")
    ## Load the dataset from file into a panda's DataFrame
    bugs = pd.read_csv('./Bug_Dataset.csv')

    ## Drop unnecessary columns
    bugs.drop('Duplicated_issue', axis=1, inplace=True)
    bugs.drop('Issue_id', axis=1, inplace=True)    
    bugs.drop('Status', axis=1, inplace=True)
    
    ## Remove rows with empty descriptions
    bugs = bugs.dropna(axis=0, inplace=False)

    ## Only keep rows where the resolution is FIXED
    bugs = bugs[bugs['Resolution'] == 'FIXED']
    bugs.drop('Resolution', axis=1, inplace=True)

    ## Reset the indexing of the dataset
    bugs = bugs.reset_index()
    bugs.drop('index',axis=1, inplace=True)
    
    if priority:
        ## Change priority to numeric values and convert the column
        bugs['Priority'] = bugs['Priority'].replace(to_replace={'P1': 0, 'P2':1, 'P3':2, 'P4':3, 'P5':4})
    else:
        ## Change priority to numeric values and convert the column
        bugs['Priority'] = bugs['Priority'].replace(to_replace={'P1': 1, 'P2':2, 'P3':3, 'P4':4, 'P5':5})

    return bugs

## Add new columns to the data set
def additional_columns(bugs):
    print("Adding additional columns...")
    ## Count the number of words in the Title and add as a new column Title_length
    titles = bugs['Title']
    titles = titles.str.split(' ')
    titles = titles.apply(lambda x: len(x))
    bugs['Title_length'] = titles
    
    ## Count the number of words in the Description and add as a new column Desc_length
    desc = bugs['Description']
    desc = desc.str.split(' ')
    desc = desc.apply(lambda x: len(x))
    bugs['Desc_length'] = desc
    
    ## Cast the series to a dataframe and export as a .txt file to run with SentiStrength
    descriptions = pd.DataFrame()
    descriptions['Desc'] = bugs['Description'].apply(lambda x: x.lower())
    descriptions['Desc'] = descriptions['Desc'].replace({'\t',';'}, '', regex=True)
    descriptions.to_csv('descriptions.txt', sep=' ', index=False)
    
    ## Read the file output from SentiStrength
    df = pd.read_csv('desc+results.txt', sep=";", names=['SentimentResults'])
    
    ## Replace "\t" with "  " and take the last 5 chars from each row which contain the positive and negative sentiment separated by "  "
    df['SentimentResults'] = df['SentimentResults'].replace('\t' , '  ', regex=True) 
    df['SentimentResults'] = df['SentimentResults'].str[-5:]
    
    ## Convert the strings into type int64
    bugs['PositiveSentiment'] = df['SentimentResults'].str[0:1].astype(str).astype(np.int64)
    bugs['NegativeSentiment'] = df['SentimentResults'].str[3:5].astype(str).astype(np.int64)
    
    return bugs
    
## Initial preparation of the resolution time column
def labels_prep(bugs, regress):
    print("Preparing labels...")

    ## Calculate resolution time in days by subtracting created and resolved time
    bugs['Created_time'] =  pd.to_datetime(bugs['Created_time'])
    bugs['Resolved_time'] = pd.to_datetime(bugs['Resolved_time'])
    time = bugs["Resolved_time"] - bugs['Created_time']

    time = time.dt.days+time.dt.seconds/86400 ## Convert the result into a floating point number of days
    
    ##Add the new column to dataset rounding to 2 decimal places
    bugs['Res_time'] = round(time,2)

    ##Drop created time and resolved time from dataset
    bugs.drop('Created_time', axis=1, inplace=True)
    bugs.drop('Resolved_time', axis=1, inplace=True)
    
    bugs = bugs[bugs['Res_time'] <= MAX_TIME] ## Remove instances where it took more than 1 year to resolve
    
    if regress: ## In the case of regression scale the column from 0 to 1
        bugs['Res_time'] = bugs['Res_time']/MAX_TIME
        
    else: ## In the case of classification split the column into the necessary number of categories
        cat = list(range(CATEGORIES))
        bugs['Res_time'] = (pd.cut(bugs['Res_time'], CATEGORIES, labels = cat)).astype(np.int64)
    
    ## Reset the indexes of the dataset
    bugs = bugs.reset_index()
    bugs.drop('index',axis=1, inplace=True)
    
    return bugs

## Initial preparation of the text features
def text_prep(bugs):
    print("Preparing textual features...")
    bugs['Title/Desc'] = bugs['Title'] + bugs['Description'] ## Join Title and Description features into one
    bugs.drop('Title', axis=1, inplace=True)
    bugs.drop('Description', axis=1, inplace=True)
    
    text = bugs['Title/Desc']
    text = text.replace("\t", " ", regex=True) ## Replace the \t occurances with empty string
    text = text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation + string.digits))) ## Remove numbers and punctuation
    
    stop = stopwords.words('english')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) ## Remove stopwords
    text = text.apply(lambda x: ' '.join([word for word in x.split() if len(word) > 3])) ## Remove words shorter than 3 characters
    text = text.apply(lambda x: x.lower()) ## Lower case the column
    bugs['Title/Desc'] = text
    
    bugs['Title/Desc'] = bugs['Title/Desc'].apply(lambda x: np.str_(x)) ## Necessary for tokenization
    
    bugs.drop('Component', axis=1, inplace=True)
    bugs.drop('Version', axis=1, inplace=True)
    
    return bugs

## Change Version and Component using OneHotEncoding to prepare for Neural Network
def onehot_prep(bugs):
    print("Performing One Hot encoding...")
    ## Removing values with low counts from Version and Component to increase model accuracy
    
    ## Remove instances where the Version occurs less than 30 times in the dataset
    versions = bugs['Version'].value_counts().keys().tolist()
    counts = bugs['Version'].value_counts().tolist()
    
    for (version, count) in zip(versions, counts):
        if count < 30:
            x = bugs[bugs['Version'] == version].index
            bugs.drop(x, inplace=True)

    ## Remove instances where the component occurs less than 100 times in the datasets
    comps = bugs['Component'].value_counts().keys().tolist()
    counts = bugs['Component'].value_counts().tolist()

    for (comp, count) in zip(comps, counts):
        if count < 100:
            x =bugs[bugs['Component'] == comp].index
            bugs.drop(x, inplace=True)

    ## OneHotEncode the Version feature
    encV = OneHotEncoder()
    
    version = bugs['Version']
    version, version_categories = version.factorize()    
    version = encV.fit_transform(version.reshape(-1,1)).toarray()    
    version = pd.DataFrame(version)
    version.columns = version_categories
    version.index = bugs.index
    
    bugs = bugs.join(version)
    
    ## OneHotEncode the Component feature
    encC = OneHotEncoder()
    
    comp = bugs['Component']
    comp, comp_categories = comp.factorize()    
    comp = encC.fit_transform(comp.reshape(-1,1)).toarray()
    comp = pd.DataFrame(comp)
    comp.columns = comp_categories
    comp.index = bugs.index
    
    bugs = bugs.join(comp)
    
    ## Reset the indexing of the data set
    bugs = bugs.reset_index()
    bugs.drop('index',axis=1, inplace=True)    
    
    return bugs

## Split the data into training, testing and further split into text features, numeric features and labels
def split_data(bugs, priority, regress):
    print("Splitting the data...")
    if priority: ## If the goal is the priority it is a classification problem and StratifiedShuffleSplit is suitable
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) ## Used to keep the same distribution
        
        for train_index, test_index in split.split(bugs, bugs['Priority']):
            bugs_train = bugs.loc[train_index]
            bugs_test = bugs.loc[test_index]
    else:
        if regress: ## In the case of regression StratifiedShuffleSplit doesn't work as it requires more than 1 instance with the same value
            bugs_train, bugs_test = train_test_split(bugs, test_size=0.2, random_state=42)
        else: ## In the case of classification use StrattifiedShuffleSplit
            print("Performing StratifiedShuffleSplit...")
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) ## Used to keep the same distribution

            for train_index, test_index in split.split(bugs, bugs['Res_time']):
                bugs_train = bugs.loc[train_index]
                bugs_test = bugs.loc[test_index]

    ## Call the column_split method to further separate the data into text features, numeric features and labels
    X_train_text, X_train_feat, y_train = column_split(bugs_train, priority)
    X_test_text, X_test_feat, y_test = column_split(bugs_test, priority)

    if (not regress) or priority: ## In the case of classification change the labels to categorical
        print("Transforming labels for softmax...")
        y_train = to_categorical(y_train, CATEGORIES)
        y_test = to_categorical(y_test, CATEGORIES)
    
    return X_train_text, X_train_feat, X_test_text, X_test_feat, y_train, y_test

## Method to split the columns of the dataset into text features, numeric features and labels
def column_split(x, priority):
    if priority: ## If the goal is priority set the Priority column as y
        y = x['Priority']
        x.drop('Priority', axis=1, inplace=True)
    else: ## If the goal is resolution time set the Res_time collumn as y
        y = x["Res_time"] 

    x.drop('Res_time', axis=1, inplace=True)    
    
    x_text = x['Title/Desc'] ## Create a separate variable for the text features
    x_feat = x.copy() ## Create a separate variable for numeric features
    x_feat.drop('Title/Desc', axis=1, inplace=True)
    return x_text, x_feat, y

## Tokenize the text features to be sequences of words, then pad to a max length
def tokenize(x_train, x_test):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    x_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=MAX_LENGTH, padding='post')
    x_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=MAX_LENGTH, padding='post')
    vocab_size = len(tokenizer.word_index) + 1 ## vocab_size is used to determine the input shape for the CNN and LSTM
    return x_train, x_test, vocab_size

## Create the architecture for the MLP
def create_mlp(dim): 
    model = Sequential() ## Use sequential model
    model.add(Dense(128, input_dim=dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.4)) ## Dropout rate to determine what percentage of nodes to activate
    model.add(Dense(32, activation='relu')) ## The final layers need to have the same number of nodes when concatenating

    return model

## Create the architecture for the CNN
def create_cnn_T(vocab_size):   
    inputShape = MAX_LENGTH
    inputs = Input(shape=(inputShape,))
    x = inputs
    
    x = Embedding(vocab_size, 128)(x) ## Add an embedding layer
    x = SpatialDropout1D(rate = 0.2)(x) ## Dropout rate to determine what percentage of nodes to activate
    x = Conv1D(128, 6, activation='relu')(x) ## 1st Convolutional Layer
    x = BatchNormalization()(x)
    x = AveragePooling1D(5)(x) ## 1st Pooling Layer
    x = Dropout(rate=0.3)(x) ## 2nd Dropout
    x = Conv1D(64, 12, kernel_regularizer=l2(0.0001), activation='relu')(x) ## 2nd Convolutional Layer
    x = BatchNormalization()(x)
    x = MaxPooling1D(3)(x) ## 2nd Pooling Layer
    x = Flatten()(x) ## Flatten the input to prepare for concatenation
    x = Dropout(rate=0.4)(x) ## 3rd Dropout
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x) ## The final layers need to have the same number of nodes when concatenating
    
    model = Model(inputs, x)
    
    return model

## Create the architecture for the LSTM
def create_lstm(vocab_size):
    inputShape = MAX_LENGTH
    inputs = Input(shape=(inputShape,))
    x = inputs
    
    x = Embedding(vocab_size, 128)(x) ## Add an embedding layer
    x = LSTM(128, dropout=0.02, recurrent_dropout=0.02)(x) ## Add an LSTM layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.4)(x) ## Dropout rate is to determine what percentage of nodes to activate
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x) ## The final layers need to have the same number of nodes when concatenating
    
    model = Model(inputs, x)
    
    return model

## Network 1 involves text and numeric features
def network_one(feat, model_comb, vocab_size, learning_rate, epochs, batches, regress):
    nn_mlp = create_mlp(feat) ## Create the MLP to train on the numeric features
    if model_comb in {1, 2, 7, 8}: ## In the case of those 4 architectures create a CNN for the text features
        nn = create_cnn_T(vocab_size) 
    elif model_comb in {4, 5, 10, 11}: ## In the case of those 4 architectures create an LSTM for the text features
        nn = create_lstm(vocab_size)
    opt = Adam(lr=learning_rate) ## Use Adam optimizer
    
    x = concatenate([nn_mlp.output, nn.output]) ## Concatinate the NN for numeric and text features
    
    if model_comb in {1, 2, 7, 8}: ## If it involves a CNN add an additional Dense layer with 64 Nodes
        x = Dense(64, activation='elu')(x) 
    x = Dense(16, activation='elu')(x)
    
    if regress: ## In the case of a regression add a final Dense layer with 1 Node and a linear activation function
        x = Dense(8, activation='elu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=[nn_mlp.input, nn.input], outputs=x) ## Train the model on the combined inputs
        ## The loss function for regression is RMSE
        model.compile(loss=root_mean_squared_error, optimizer=opt)
    else: ## In the case of a classification add a final Dense layer with same number of nodes as categories
        x = Dense(CATEGORIES, name='output', activation='softmax')(x) ## The activation function is Softmax
        model = Model(inputs=[nn_mlp.input, nn.input], outputs=x) ## Train the model on teh combined inputs
        ##The loss function for classification is categorical crossentropy
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
    return model

## Network 2 which only involves text features
def network_two(model_comb, vocab_size, learning_rate, epochs, batches, regress):
    if model_comb in {3, 9}: ## Those 2 architectures involve training text features on a CNN
        nn = create_cnn_T(vocab_size)
    elif model_comb in {6, 12}: ## Those 2 architectures involve training text features on an LSTM
        nn = create_lstm(vocab_size)
    opt = Adam(lr=learning_rate) ## Use Adam optimizer
    
    x = Dense(16, activation='elu')(nn.output)
    
    if regress: ## In case of regression add a final Dense layer with 1 node and linear activation
        x = Dense(8, activation='elu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=nn.input, outputs=x)
        ## The loss function for regression is RMSE
        model.compile(loss=root_mean_squared_error, optimizer=opt)
    else: ## In case of classification add a final Dense layer with the same number of nodes as the number of categories
        x = Dense(CATEGORIES, name='output', activation='softmax')(x) ## The activation function is softmax
        model = Model(inputs=nn.input, outputs=x)
        ## The loss function for classification is categorical crossentropy
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
     
    return model
     
## Method to test the trained model against the testing set
def test_model(X_test_text, X_test_feat, y_test, model, regress, model_comb):
    if model_comb in {1,2,4,5,7,8,10,11}: ## Those architectures include both text and numerical features
        if regress: ##If it is a regression problem calculate the RMSE
            preds = model.predict([X_test_feat, X_test_text], verbose=1)
            mse = mean_squared_error(y_test, preds.flatten())
            rmse = np.sqrt(mse)
        
            print("Accuracy for Regression using RMSE: {}.".format(round(rmse, 4)))            
        else: ##Else calculate the accuracy percentage
            score = model.evaluate([X_test_feat, X_test_text], y_test, verbose=1)            

            print("Accuracy: {}%.".format(round((score[1]*100),2)))
    else: ## The remaining 4 architectures include only text features
        if regress: ##If it is a regression problem calculate the RMSE
            preds = model.predict(X_test_text, verbose=1)
            mse = mean_squared_error(y_test, preds.flatten())
            rmse = np.sqrt(mse)

            print("Accuracy for regression using RMSE: {}.".format(round(rmse, 4)))            
        else: ##Else calculate the accuracy percentage
            score = model.evaluate(X_test_text, y_test, verbose=1)            

            print("Accuracy for classification: {}%.".format(round((score[1]*100), 2)))

## Restore a previously saved model
def restore_model(checkpoint_path):
    print("Restoring selected model...")
    model_comb = np.int(checkpoint_path.split("-")[1])
    regress = np.bool(checkpoint_path.split("-")[2])
    learning_rate = np.float32(checkpoint_path.split("-")[3])
    epochs = np.int(checkpoint_path.split("-")[4])
    batches = np.int(checkpoint_path.split("-")[5])
    restore = True
    X_train_text, X_train_feat, X_test_text, X_test_feat, y_train, y_test, model = main(model_comb, regress, learning_rate, epochs, batches, restore, checkpoint_path)
    return X_train_text, X_train_feat, X_test_text, X_test_feat, y_train, y_test, model

def plot_model_training(history, regress):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
        
    if not regress: 
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('iteration')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

## Method to create a baseline model to compare to the Neural Networks
def baseline(X_train, y_train, X_test, y_test, regress):
    if regress: ## In case of regression create and evaluate a Linear Regression model
        print("Establishing a baseline regression model using Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds.flatten())
        rmse = np.sqrt(mse)
        
        print("Test accuracy for the baseline model using RMSE: {}.".format(round(rmse,4)))
    else: ## In case of classification create and evaluate a Decision Tree Classifier
        print("Establishing a baseline classification model using Decision Tree Classifier...")
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)            

        print("Test accuracy for the baseline classification model: {}%.".format(round((score*100),2)))
    
    return model

## Custom function for Root Mean Squared Error
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

## Checks if the input parameter is a float
def check_param_is_float(param, value):
    try:
        value = float(value)
    except:
        print("{} must be float".format(param))
        quit(1)
    return value    

## Checks if the input parameter is an integer
def check_param_is_int(param, value):
    try:
        value = int(value)
    except: 
        print("{} must be integer".format(param))
        quit(1)
    return value

## Checks if the input parameter is a boolean
def check_param_is_bool(param, value):
    try:
        value = bool(value)
    except:
        print("{} must be boolean".format(param))
        quit(1)
    return value

#Combination 1:(Res_time) A combination of CNN and MLP (including Version and Component)
#Combination 2:(Res_time) A combination of CNN and MLP (without Version and Component)
#Combination 3:(Res_time) A CNN only for the text features
#Combination 4:(Res_time) A combination of LSTM and MLP (including Version and Component)
#Combination 5:(Res_time) A combination of LSTM and MLP (without Version and Component)
#Combination 6:(Res_time) An LSTM only for the text features
#Combination 7:(Priority) A combination of CNN and MLP (including Version and Component)
#Combination 8:(Priority) A combination of CNN and MLP (without Version and Component)
#Combination 9:(Priority) A CNN only for the text features
#Combination 10(Priority): A combination of LSTM and MLP (including Version and Component)
#Combination 11(Priority): A combination of LSTM and MLP (without Version and Component)
#Combination 12(Priority): An LSTM only for the text features

## model_comb, regress, learning_rate, epochs, batches
#bugs, X_train_text, X_train_feat, X_test_text, X_test_feat, Y_train, Y_test, model = main(model_comb=model_numb, regress=regress, learning_rate=learning_rate, epochs=epochs, batches=batches)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("model_comb", help="Model Selection")
    arg_parser.add_argument("regress", help="Regression or Classification")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    args = arg_parser.parse_args()

    model_comb = check_param_is_int("model_comb", args.model_comb)
    regress = check_param_is_bool("regress", args.regress)
    learning_rate = check_param_is_float("learning_rate", args.learning_rate)
    epochs = check_param_is_int("epochs", args.iterations)
    batches = check_param_is_int("batches", args.batches)

    main(model_comb, regress, learning_rate, epochs, batches)
    