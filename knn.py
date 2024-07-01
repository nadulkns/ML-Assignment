import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("DataSet/data3.csv")

# create a new column and classify data as Class C1 and C2
df.loc[:99, 'Class'] = 'C1'
df.loc[99:199, 'Class'] = 'C2'


# Create a separate dataframe for class C1 and C2
df0 = df[:99]
df1 = df[99:199]
df2 = df[199:]

# plot data in a scatter plot
plt.xlabel('Col1')
plt.ylabel('Col2')
plt.title('Scatter Plot')
plt.scatter(df0['Col1'], df0['Col2'], color='green',
            marker='+', label='Class C1')
plt.scatter(df1['Col1'], df1['Col2'], color='blue',
            marker='.', label='Class C2')
plt.scatter(df2['Col1'], df2['Col2'], color='red',
            marker='*', label='Predicted')
plt.legend(loc='upper left')
plt.show()

# Assign values for X and y variables
X = df.drop(['Class'], axis='columns')
y = df.Class

# Test and Train data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.09, random_state=None, shuffle=False)

# Create KNN Model


def knn_model(X_train, X_test, y_train, y_test, k, metric):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)
    predict = knn.predict(X_test)
    score = knn.score(X_test, predict)

    d = pd.DataFrame({'Sample': X_test.index, 'Prediction': predict})
    print(d)

    print(f"\nAccuracy of Distances_type={metric} with k={k} is {score}")
    print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n")
    return 0


knn_model(X_train, X_test, y_train, y_test, 1, 'euclidean')
knn_model(X_train, X_test, y_train, y_test, 3, 'euclidean')
print("\n")

knn_model(X_train, X_test, y_train, y_test, 1, 'manhattan')
knn_model(X_train, X_test, y_train, y_test, 3, 'manhattan')
print("\n")

knn_model(X_train, X_test, y_train, y_test, 1, 'cosine')
knn_model(X_train, X_test, y_train, y_test, 3, 'cosine')
print("\n")

knn_model(X_train, X_test, y_train, y_test, 1, 'chebyshev')
knn_model(X_train, X_test, y_train, y_test, 3, 'chebyshev')
print("\n")
