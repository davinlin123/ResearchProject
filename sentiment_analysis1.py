import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
# we import the many libraries we will be using

# Day 1
#read the file and clean it, setting it to lowercase and removing all punctuation
train_df = pd.read_csv("Train.csv")
print(train_df.head())
print(train_df.columns)

train_df['text'] = train_df['text'].str.lower()
train_df['text'] = train_df['text'].str.replace(r'[^\w\s]', '', regex=True)
train_df['text'] = train_df['text'].str.strip()

print(train_df['text'].head())


# Day 2
#check for null and further clean it to only letters and spaces
print(train_df.isnull().sum())
print(train_df['label'].value_counts())
train_df['clean_text'] = train_df['text'].str.lower()
train_df['clean_text'] = train_df['clean_text'].str.replace(r'[^a-z\s]', '', regex=True)
print(train_df[['text', 'clean_text']].head())


# Day 3
#convert the words to numbers using vectorization and also uses max_features to keep the top 5000 most important words
X = train_df['clean_text']
y = train_df['label']
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)
X_vectors = vectorizer.fit_transform(X)
print(X_vectors.shape)

# Day 4 & 5
#reads the data and then uses 75% of it for training while the other 25% is to test accuracy
model = LogisticRegression(max_iter=1000)
model.fit(X_vectors, y)
print(model.score(X_vectors,y))


X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, train_size = .75, random_state = 67)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Test Accuracy", accuracy)

# Day 6
#we compare the accuracies and use confusion & classification to see error patterns
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
#actual vs prediction
report = classification_report(y_test, y_pred)
print(report)

# Day 7
#we try a different model for the accuracy and see how it differs
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_accuracy = nb_model.score(X_test, y_test)
print("NB Accuracy", nb_accuracy)
#classification for nb
nb_y_pred = nb_model.predict(X_test)
nb_report = classification_report(y_test, nb_y_pred)
print(nb_report)

# Day 8
#comparing the ngrams vs min_df accuracy
model = LogisticRegression(max_iter=1000)
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)
)
X_vectors = vectorizer.fit_transform(train_df['clean_text'])
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, train_size = .75, random_state = 67)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Test Accuracy (ngrams):", accuracy)

model = LogisticRegression(max_iter=1000)
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    min_df=10
)
X_vectors = vectorizer.fit_transform(train_df['clean_text'])
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, train_size = .75, random_state = 67)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Test Accuracy (min_df)", accuracy)
