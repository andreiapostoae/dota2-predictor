import pandas
import patsy
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn import metrics, cross_validation 
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def data_size_response(model,trX,teX,trY,teY,score_func,prob=True,n_subsets=20):

    train_errs,test_errs = [],[]
    subset_sizes = np.exp(np.linspace(3,np.log(trX.shape[0]),n_subsets)).astype(int)

    for m in subset_sizes:
        model.fit(trX[:m],trY[:m])
        if prob:
            train_err = score_func(trY[:m],model.predict_proba(trX[:m]))
            test_err = score_func(teY,model.predict_proba(teX))
        else:
            train_err = 1 - score_func(trY[:m],model.predict(trX[:m]))
            test_err = 1 - score_func(teY,model.predict(teX))
        print "training error: %.3f test error: %.3f subset size: %.3f" % (train_err,test_err,m)
        train_errs.append(train_err)
        test_errs.append(test_err)

    return subset_sizes,train_errs,test_errs

def plot_response(subset_sizes,train_errs,test_errs):

    plt.plot(subset_sizes,train_errs,lw=2)
    plt.plot(subset_sizes,test_errs,lw=2)
    plt.legend(['Training Error','Test Error'])
    plt.xscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Error')
    plt.title('Model response to dataset size')
    plt.show()
filename = 'games_processed.csv'

names = ['match_id']
for i in range(228):
	names.append("hero" + str(i + 1))
names.append('win')

#read data
data = pandas.read_csv(filename)

#format data
m = data.as_matrix()

y = m[:, -1]
X = m[:, 1:231]
print X[0]

print y.mean()
print X.shape
print y.shape

#initialize structs
y = np.ravel(y)


#train and test results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha", np.logspace(-7, 3, 3))
#print train_scores
#print valid_scores

model2 = LogisticRegression()
model2.fit(X_train, y_train)
predicted = model2.predict(X_test)
probs = model2.predict_proba(X_test)

print metrics.accuracy_score(y_test, predicted)
print metrics.roc_auc_score(y_test, probs[:, 1])

joblib.dump(model2, 'model2.pkl')


#print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model2, X_train, y_train, cv=20, scoring='roc_auc'))
'''model = model2
score_func = metrics.accuracy_score
response = data_size_response(model,X_train,X_test,y_train,y_test,score_func,prob=False)
plot_response(*response)
'''
'''
hero_ids = [39, 71, 3, 76, 4, 25, 14, 26, 36]

with open('model2.pkl', 'rb') as f:
	model = joblib.load(f)

new_list = []
for i in range(228):
	new_list.append(0)

for i in range(9):
	if(i < 5):
		new_list[hero_ids[i] - 1] = 1
	else:
		new_list[hero_ids[i] + 113] = 1

dictionary = {}
for i in range(114):
	new_list[i] = 1
	res = model.predict_proba(np.array(new_list).reshape(1, -1))
	dictionary[i + 1] = res[0][0] * 100
	new_list[i] = 0

import operator
sorted_dict = sorted(dictionary.items(), key = operator.itemgetter(1))

for (key, value) in sorted_dict:
	print("%d: %.3f" % (key, value))
'''