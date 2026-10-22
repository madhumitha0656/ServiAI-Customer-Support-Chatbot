import random #used to generate a random number
import json #used to pass json used to read and write file
import pickle #for serialization and deserialization of a Python object structure
import numpy as np #for numerical operations
import tensorflow as tf #to create machine or deep learning models
import nltk #natial language toolkit for text processing
from nltk.stem import WordNetLemmatizer #used to reduce words to their base or root form
lemmetizer = WordNetLemmatizer() #creating an instance of the WordNetLemmatizer class
intents = json.loads(open(r'C:\Users\vijay\OneDrive - Bharatiya Vidya Bhavan Bhavans Sri RamaKrishna Vidyalaya\nlp_bot\intents.json').read()) #loading the intents.json file and reading its content
words=[] #list to store the words
classes = [] #list to store the classes
documents = [] #list to store the documents
ignore_letters = ['?', '!', '.', ','] #list of characters to ignore
for intent in intents['intents']: 
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #tokenizing the pattern into words
        words.extend(word_list) #adding the words to the words list
        documents.append((word_list, intent['tag'])) #adding the word list and tag to the documents list
        if intent['tag'] not in classes:
            classes.append(intent['tag']) #adding the tag to the classes list
words = [lemmetizer.lemmatize(word) for word in words if word not in ignore_letters] #lemmatizing the words and converting them to lowercase
words = sorted((set(words))) #removing duplicates and sorting the words
classes = sorted(set(classes)) #removing duplicates and sorting the classes
pickle.dump(words, open("words.pkl" , 'wb')) #saving the words list to a pickle file
pickle.dump(classes, open("classes.pkl" , 'wb')) #saving the classes list to
training = [] #list to store the training data
output_empty = [0] * len(classes) #creating an empty output list
for document in documents:
    bag = [] #list to store the bag of words
    word_patterns = document[0] #getting the word patterns from the document
    word_patterns = [lemmetizer.lemmatize(word.lower()) for word in word_patterns] #lemmatizing the word patterns and converting them to lowercase
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) #creating a bag of words
    output_row = list(output_empty) #creating a copy of the empty output list
    output_row[classes.index(document[1])] = 1 #setting the index of the class to 1
    training.append(bag+output_row) #adding the bag of words and output row to the training list
random.shuffle(training) #shuffling the training data
training = np.array(training) #converting the training list to a numpy array
train_x = training[:, :len(words)] #getting the input data
train_y = training[:, len(words):] #getting the output data,helps in getting the data in one hot encoder form(binary form)
model = tf.keras.Sequential() #creating a sequential model
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),),activation="relu")) #adding a dense layer with 128 neurons and relu activation function
model.add(tf.keras.layers.Dropout(0.5)) #adding a dropout layer with 0.5 dropout rate
model.add(tf.keras.layers.Dense(64, activation="relu")) #adding a dense layer with 64 neurons and relu activation function
model.add(tf.keras.layers.Dropout(0.5)) #adding a dropout layer with 0.5 dropout rate
model.add(tf.keras.layers.Dense(len(train_y[0]), activation="softmax")) #adding a dense layer with the number of classes and softmax activation function

#sgd=stochastic gradient descent and optimization algorithm that helps the model learn
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True) #creating an SGD optimizer with learning rate, decay, momentum and nesterov parameters
#Nesterov looks ahead before it jumps, so it can hit the brakes if it's about to overshoot the finish line.
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])


hist=model.fit(
    np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1
)

model.save("chatbot_model.h5", hist) #saving the model to a file
print("done") #printing done when the model is trained and saved