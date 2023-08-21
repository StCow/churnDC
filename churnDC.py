
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz

# read / clean data
data = pd.read_csv()
data.drop(['State', 'Voice'], axis = 1, inplace = True)

# define variables
y = data['Churn'].astype['int']
x = data.drop('Churn', axis = 1)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3, random_state = 17)

# first tree
first_tree = DecisionTreeClassifier(random_state = 17)

# cross validation 5 times
cross_val_score(first_tree, x_train, y_train, cv = 5)

# calculating average value
np.mean(cross_val_score(first_tree, x_train, y_train, cv = 5))

# first KNN
first_knn = KNeighborsClassifier

np.mean(cross_val_score(first_knn, x_train, y_train, cv = 5))

# tree grid
tree_params = {'max_depth': np.arange(1, 11)}
tree_grid = GridSearchCV(first_tree, tree_params, cv = 5, n_jobs = -1)
tree_grid.fit(x_train, y_train)
tree_grid.best_params_

# tune k for KNN
knn_params = {'n_neighbors': [1, 2, 3, 4] + list(range(50, 100, 10))}
knn_grid = GridSearchCV(first_knn, knn_params, cv = 5)
knn_grid.fit(x_train, y_train)
knn_grid.best_score_, knn_grid.best_params_

# choose the best decision tree
tree_grid.best_estimator_

# check
tree_grid.pred = tree_grid.predict(x_valid)
accuracy_score(y_valid, tree_valid_prod)

# decision tree graph
graph = Source(tree.export_graphviz(clf, out_file = none, feature_names = list(x), class_names=['Negative', 'Positive'], filled = True))
display(SVG(graph.pipe(format='svg')))
