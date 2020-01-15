from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import entropy

categories = ['rec.motorcycles', 'rec.sport.baseball',
                'comp.graphics', 'sci.space',
                'talk.politics.mideast']
remove = ("headers", "footers", "quotes")
ng5_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
ng5_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

## Exploring Data ##
print("\n".join(ng5_train.data[0].split("\n")))
print(ng5_train.target_names[ng5_train.target[0]])

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

# Define function which updates datasets
def update(seed, unlabeled, ind):
     seed.data = seed.data + [unlabeled.data[i] for i in ind]
     seed.target = np.append(seed.target,unlabeled.target[ind])
     unlabeled.data = [unlabeled.data[i] for i in range(len(unlabeled.data)) if i not in ind]
     unlabeled.target = np.delete(unlabeled.target, ind)
     return seed, unlabeled

def update_accuracy(seed, unlabeled, accuracy):
     global text_clf
     # Train the clasifier
     text_clf.fit(seed.data, seed.target)
     # predicted classes and corresponding probabilities
     predicted = text_clf.predict(unlabeled.data)
     predicted_proba = text_clf.predict_proba(unlabeled.data)
     accuracy.append(np.mean(predicted == unlabeled.target))
     return accuracy, predicted_proba 
     
length = len(ng5_train.data)
seed_uncert = seed_margin = seed_entropy = Bunch(data=ng5_train.data[0:1500], target=ng5_train.target[0:1500])
unlabeled_uncert = unlabeled_margin = unlabeled_entropy = Bunch(data=ng5_train.data[1500:length-1], target=ng5_train.target[1500:length-1])

# number of samples
n = 100
accuracy_uncert = accuracy_margin = accuracy_entropy = []

# Start the loop
while len(unlabeled_uncert.data) > 0:
     ## Least Confidence (aka. Uncertainty) Strategy
     accuracy_uncert, predicted_proba = update_accuracy(seed_uncert, unlabeled_uncert, accuracy_uncert)
     uncertainty = 1 - predicted_proba.max(axis=1)
     ## Margin Sampling
     #accuracy_margin, predicted_proba = update_accuracy(seed_margin, unlabeled_margin, accuracy_margin)
     #part = np.partition(-predicted_proba, 1, axis=1)
     #margin = - part[:, 0] + part[:, 1]
     ## Entropy based
     #accuracy_entropy, predicted_proba = update_accuracy(seed_entropy, unlabeled_entropy, accuracy_entropy)
     #entropy = entropy(predicted_proba.T)

     if len(unlabeled_uncert.data) > n : 
          # index of top n uncertainty score
          ind_uncert = np.argpartition(uncertainty, -n)[-n:]
          # index of n min margin score
          #ind_margin = np.argpartition(margin, n)[:n]
          # index of top n entropy score
          #ind_entropy = np.argpartition(entropy, -n)[-n:]

          seed_uncert, unlabeled_uncert = update(seed_uncert, unlabeled_uncert, ind_uncert)
          #seed_margin, unlabeled_margin = update(seed_margin, unlabeled_margin, ind_margin)
          #seed_entropy, unlabeled_entropy = update(seed_entropy, unlabeled_entropy, ind_entropy)
     else :
          # seed.data = seed.data + unlabeled.data
          # seed.target = np.append(seed.target, unlabeled.target)
          # text_clf.fit(seed.data, seed.target)
          unlabeled_uncert.data = unlabeled_margin = unlabeled_entropy = []

