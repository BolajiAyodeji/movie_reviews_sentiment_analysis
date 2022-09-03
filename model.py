import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle

df_review = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df_review

df_positive = df_review[df_review['sentiment']=='positive'][:40000]
df_negative = df_review[df_review['sentiment']=='negative'][:10000]

df_review_imb = pd.concat([df_positive, df_negative])

sns.set_style('darkgrid')
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('font', size=14)

colors = sns.color_palette('deep')

plt.figure(figsize=(10,8), tight_layout=True)
plt.bar(x=['Positive', 'Negative'], height=df_review_imb.value_counts(['sentiment']), color=colors[:2])
plt.title('Sentiment')
plt.savefig('sentiment.png')

plt.show()

#Balancing the data set

rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review_imb[['review']], df_review_imb['sentiment'])

df_review_bal

print(df_review_imb.value_counts('sentiment'))
print('---------------')
print(df_review_bal.value_counts('sentiment'))

#Splitting the dataset into train and test data sets

train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)

train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

#Text representation to numerical vectors using bag of words

tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)

train_x_vector

test_x_vector = tfidf.transform(test_x)

pd.DataFrame.sparse.from_spmatrix(train_x_vector,index=train_x.index,columns=tfidf.get_feature_names())

#Save the vectorizer

import pickle

pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

#Testing the model using four classification models

#Support Vector Machine (SVM)

svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

#Decision Tree

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

#Naive Bayes

gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

#Logistic Rgression

log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

#Models evaluation

#Mean accuracy

print('Support vector machine:', svc.score(test_x_vector, test_y))
print('Decision tree:', dec_tree.score(test_x_vector, test_y))
print('Naive bayes:', gnb.score(test_x_vector.toarray(), test_y))
print('Logistic regression:', log_reg.score(test_x_vector, test_y))

#F1 score (using the SVC model)

f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None)

#Classification report (using the SVC model)

print(classification_report(test_y,svc.predict(test_x_vector), labels=['positive', 'negative']))

#Confusion matrix (using the SVC model)

conf_mat = confusion_matrix(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])

conf_mat

#Test model with a new review

review = ["Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause"]
new_review = tfidf.transform(review)
svc.predict(new_review)

#Model optimization using GridSearchCV

parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc,parameters, cv=5)

svc_grid.fit(train_x_vector, train_y)

print(svc_grid.best_params_)
print(svc_grid.best_estimator_)

#Save model with Pickle, load model, and test

filename = 'movie_reviews_sentiment_analysis.pkl'

pickle.dump(svc_grid, open(filename, 'wb'))

#Load and test saved model
loaded_model = pickle.load(open(filename, 'rb'))

review = ["This is pretty much the worse movie I have ever watched. It's completely thrash!"]
new_review = tfidf.transform(review)

result = loaded_model.predict(new_review)
print(result)
