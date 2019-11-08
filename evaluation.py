# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 00:54:40 2019

@author: Mihail Mihaylov
"""
import collections
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping

bugs.info()

res_time_test_d = Y_test.describe()
fast = res_time_test_d["50%"]
slow = res_time_test_d['75%']

fast_test = bugs_test[bugs_test['Res_time']<=fast]
slow_test = bugs_test[bugs_test['Res_time']>=slow]
medium_test = bugs_test[bugs_test['Res_time']<slow][bugs_test['Res_time']>fast]

fast_test_text, fast_test_feat, fast_test_labels = column_split(fast_test, priority)
slow_test_text, slow_test_feat, slow_test_labels = column_split(slow_test, priority)
medium_test_text, medium_test_feat, medium_test_labels = column_split(medium_test, priority)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(bugs_train["Title/Desc"])
vocab_size = len(tokenizer.word_index) + 1 ## vocab_size is used to determine the input shape for the CNN and LSTM
fast_test_text_t = pad_sequences(tokenizer.texts_to_sequences(fast_test_text), maxlen=MAX_LENGTH, padding='post')
slow_test_text_t = pad_sequences(tokenizer.texts_to_sequences(slow_test_text), maxlen=MAX_LENGTH, padding='post')
medium_test_text_t = pad_sequences(tokenizer.texts_to_sequences(medium_test_text), maxlen=MAX_LENGTH, padding='post')

fast_eval = test_model(fast_test_text_t, fast_test_feat, fast_test_labels, model, regress, model_numb)
slow_eval = test_model(slow_test_text_t, slow_test_feat, slow_test_labels, model, regress, model_numb)
medium_eval = test_model(medium_test_text_t, medium_test_feat, medium_test_labels, model, regress, model_numb)

res_time_train_d = Y_train.describe()
fast_t = res_time_train_d["50%"]
slow_t = res_time_train_d['75%']

fast_train = bugs_train[bugs_train['Res_time']<=fast]
slow_train = bugs_train[bugs_train['Res_time']>=slow]
medium_train = bugs_train[bugs_train['Res_time']<slow][bugs_train['Res_time']>fast]

fast_train.describe() #14493
slow_train.describe() #7442
medium_train.describe() #7362

fast_train = fast_train.sample(7402) #7402

bugs_train_mod = pd.concat([fast_train, slow_train, medium_train], axis=0)
bugs_train_mod = shuffle(bugs_train_mod)
bugs_train_text_mod, bugs_train_feat_mod, bugs_train_labels_mod = column_split(bugs_train_mod, priority)

bugs_train_text_mod_t = pad_sequences(tokenizer.texts_to_sequences(bugs_train_text_mod), maxlen=MAX_LENGTH, padding='post')


checkpoint_path = "./Models/bugs-{}-{}-{}-{}-{}-Opt.ckpt".format(model_numb, regress, learning_rate, epochs, batches)
cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=0, period=1)


## TO TRAIN THE MODEL
feat = bugs_train_feat_mod.shape[1]
model_opt = network_one(feat, model_numb, vocab_size, learning_rate, epochs, batches, regress)    
X = [bugs_train_feat_mod, bugs_train_text_mod_t]


history1 = model_opt.fit(X, bugs_train_labels_mod, batch_size=batches, epochs=3, verbose=1,
                  validation_split=0.1, shuffle='batch', callbacks = [cp_callback])        
plot_model_training(history1, regress)
        

r_path_opt = "./Models/bugs-{}-{}-{}-{}-{}-Opt.ckpt".format(model_comb, regress, learning_rate, epochs, batches)

model_opt.load_weights(r_path)
fast_eval_opt = test_model(fast_test_text_t, fast_test_feat, fast_test_labels, model_opt, regress, model_numb)
slow_eval_opt = test_model(slow_test_text_t, slow_test_feat, slow_test_labels, model_opt, regress, model_numb)
medium_eval_opt = test_model(medium_test_text_t, medium_test_feat, medium_test_labels, model_opt, regress, model_numb)

test_opt = test_model(X_test_text, X_test_feat, Y_test, model_opt, regress, model_numb)

###################################################################################

y_train_num = bugs_train["Priority"]
y_test_num = bugs_test["Priority"]

y_train_num.value_counts()
y_test_num.value_counts()

preds = model.predict(X_test_text, verbose=1)

preds_class = np.argmax(preds, axis=1)
preds_class = preds_class.tolist()


collections.Counter(preds_class)
y_test_num.value_counts()

confusion_matrix(y_test_num, preds_class)

F1 = f1_score(y_test_num, preds_class, average='macro')

precision = average_precision_score(y_test_num, preds_class)

bugs_train.info()
p1 = bugs_train[bugs_train['Priority']==0]
len(p1)
p2 = bugs_train[bugs_train['Priority']==1]
len(p2)
p3 = bugs_train[bugs_train['Priority']==2]
len(p3)
p4 = bugs_train[bugs_train['Priority']==3]
len(p4)
p5 = bugs_train[bugs_train['Priority']==4]
len(p5)

p3 = p3.sample(2511)
bugs_train_mod = pd.concat([p1, p2, p3, p4, p5], axis=0)
bugs_train_mod = shuffle(bugs_train_mod)

bugs_train_text_mod, bugs_train_feat_mod, bugs_train_labels_mod = column_split(bugs_train_mod, priority)

bugs_train_labels_mod_cat = to_categorical(bugs_train_labels_mod, CATEGORIES)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(bugs_train_text_mod)
vocab_size = len(tokenizer.word_index) + 1 ## vocab_size is used to determine the input shape for the CNN and LSTM
bugs_train_text_mod_t = pad_sequences(tokenizer.texts_to_sequences(bugs_train_text_mod), maxlen=MAX_LENGTH, padding='post')

checkpoint_path_11 = "./Models/bugs-{}-{}-{}-{}-{}-Opt.ckpt".format(model_numb, regress, learning_rate, epochs, batches)
cp_callback_11 = ModelCheckpoint(checkpoint_path_11, save_weights_only=True, verbose=0, period=1)

checkpoint_path_12 = "./Models/bugs-{}-{}-{}-{}-{}-Opt.ckpt".format(12, regress, learning_rate, epochs, batches)
cp_callback_12 = ModelCheckpoint(checkpoint_path_12, save_weights_only=True, verbose=0, period=1)

va_callback = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=10)

##Model 12
model_comb=12
model_12_opt = network_two(model_comb, vocab_size, learning_rate, epochs, batches, regress)
X_12 = bugs_train_text_mod_t

history12 = model_opt.fit(X_12, bugs_train_labels_mod_cat, batch_size=batches, epochs=100, verbose=1,
                  validation_split=0.1, shuffle='batch', callbacks = [cp_callback_11, va_callback])        
plot_model_training(history12, regress)

##Model 11
model_comb=11
feat = bugs_train_feat_mod.shape[1]
model_11_opt = network_one(feat, model_comb, vocab_size, learning_rate, epochs, batches, regress)
X_11 = [bugs_train_feat_mod, bugs_train_text_mod_t]

history11 = model_11_opt.fit(X_11, bugs_train_labels_mod_cat, batch_size=batches, epochs=100, verbose=1,
                  validation_split=0.1, shuffle='batch', callbacks = [cp_callback_12, va_callback])        
plot_model_training(history11, regress)

###12
y_test_num.value_counts()

preds_12 = model_12_opt.predict(X_test_text, verbose=1)

preds_class_12 = np.argmax(preds_12, axis=1)
preds_class_12 = preds_class_12.tolist()

confusion_matrix(y_test_num, preds_class_12)

F1_12 = f1_score(y_test_num, preds_class_12, average='macro')

collections.Counter(preds_class_12)

###11
y_test_num.value_counts()

preds_11 = model_11_opt.predict([X_test_feat, X_test_text], verbose=1)

preds_class_11 = np.argmax(preds_11, axis=1)
preds_class_11 = preds_class_11.tolist()

confusion_matrix(y_test_num, preds_class_11)

F1_11 = f1_score(y_test_num, preds_class_11, average='macro')
F1

collections.Counter(preds_class_11)

F1
F1_11
F1_12
model_comb = 12
test_model(X_train_text, X_train_feat, Y_train, model_12_opt, regress, model_comb)
test_model(X_test_text, X_test_feat, Y_test, model_12_opt, regress, model_comb)


