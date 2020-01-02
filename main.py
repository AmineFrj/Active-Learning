from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
import numpy as np

categories = ['rec.motorcycles', 'rec.sport.baseball',
                'comp.graphics', 'sci.space',
                'talk.politics.mideast']
remove = ("headers", "footers", "quotes")
ng5_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
ng5_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

## Exploring Data ##
print("\n".join(ng5_train.data[0].split("\n")))
print(ng5_train.target_names[ng5_train.target[0]])

##  Third step : Select n samples from non labeled data U  ##
## Create a pipeline to make it simpler
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
])

#del(ng5_train.data[0])
#ng5_train.target = np.delete(ng5_train.target,0)

#### Start the loop process #####
from sklearn.utils import Bunch

length = len(ng5_train.data)
seed = Bunch(data=ng5_train.data[0:1500], target=ng5_train.target[0:1500])
unlabeled = Bunch(data=ng5_train.data[1500:length-1], target=ng5_train.target[1500:length-1])


# number of samples
n = 100
accuracy = []

while len(unlabeled.data) > 0:
     text_clf.fit(seed.data, seed.target)

     # predicted classes and corresponding probabilities
     predicted = text_clf.predict(unlabeled.data)
     predicted_proba = text_clf.predict_proba(unlabeled.data)
     accuracy.append(np.mean(predicted == unlabeled.target))

     ## Least Confidence (aka. Uncertainty) Strategy
     uncertainty = 1 - predicted_proba.max(axis=1)
     #uncertainty.size

     if len(unlabeled.data) > n : 
          # index of top n uncertainty score
          ind = np.argpartition(uncertainty, -n)[-n:]
          #uncertainty[ind]

          seed.data = seed.data + [unlabeled.data[i] for i in ind]
          seed.target = np.append(seed.target,unlabeled.target[ind])
          unlabeled.data = [unlabeled.data[i] for i in range(len(unlabeled.data)) if i not in ind]
          #unlabeled.target = [unlabeled.target[i] for i in range(len(unlabeled.target)) if i not in ind]
          unlabeled.target = np.delete(unlabeled.target, ind)
     else :
          seed.data = seed.data + unlabeled.data
          seed.target = np.append(seed.target, unlabeled.target)
          text_clf.fit(seed.data, seed.target)
          unlabeled.data = []
     
