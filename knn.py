import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sys


df = pd.read_csv("DataSet/data3.csv")

# create a new column and classify data as Class C1 and C2
df.loc[:99, 'Class'] = 'C1'
df.loc[99:199, 'Class'] = 'C2'


# Create  separate dataframes for class C1,C2 and predicted
df0 = df[:99]
df1 = df[99:199]
df2 = df[199:]


def plot():  # plot data in a scatter plot
    plt.xlabel('Col1')
    plt.ylabel('Col2')
    plt.title('Scatter Plot')
    plt.scatter(df0['Col1'], df0['Col2'], color='green',
                marker='+', label='Class C1')
    plt.scatter(df1['Col1'], df1['Col2'], color='blue',
                marker='.', label='Class C2')
    plt.scatter(df2['Col1'], df2['Col2'], color='red',
                marker='*', label='to Predict')
    plt.legend(loc='upper left')
    plt.show()
    main_menu()


# Assign values for X and y variables
X = df.drop(['Class'], axis='columns')
y = df.Class

# Train and Test data
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

# Create Weighted KNN Model


def weighted_knn_model(X_train, X_test, y_train, y_test, k, metric, weights):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    knn.fit(X_train, y_train)
    predict = knn.predict(X_test)
    score = knn.score(X_test, predict)

    d = pd.DataFrame({'Sample': X_test.index, 'Prediction': predict})
    print(d)

    print(f"\nAccuracy of Distances_type={metric} with k={k} is {score}")
    print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n")
    return 0


def a_b():  # Answers for question a and b
    knn_model(X_train, X_test, y_train, y_test, 1, 'euclidean')
    knn_model(X_train, X_test, y_train, y_test, 3, 'euclidean')
    print("\n")

    main_menu()


def c_d():  # Answers for question c and d
    knn_model(X_train, X_test, y_train, y_test, 1, 'manhattan')
    knn_model(X_train, X_test, y_train, y_test, 3, 'manhattan')
    print("\n")

    knn_model(X_train, X_test, y_train, y_test, 1, 'cosine')
    knn_model(X_train, X_test, y_train, y_test, 3, 'cosine')
    print("\n")

    knn_model(X_train, X_test, y_train, y_test, 1, 'chebyshev')
    knn_model(X_train, X_test, y_train, y_test, 3, 'chebyshev')
    print("\n")
    main_menu()


def e():  # Answers for question e
    print("***weighted K nearest neighbor algorithm***\n")
    weighted_knn_model(X_train, X_test, y_train,
                       y_test, 1, 'euclidean', 'distance')
    weighted_knn_model(X_train, X_test, y_train, y_test,
                       3, 'euclidean', 'distance')
    main_menu()


def main_menu():
    print("    Machine Learning - Assignment 1    AS2021402\n")

    print("Press 1 : Plot data")
    print("Press 2 : Answers for Question a & b")
    print("Press 3 : Answers for Question c & d")
    print("Press 4 : Answers for Question e")
    print("press 5 : Exit")

    choice = input("\nEnter your Choice :")
    print("\n")

    match choice:
        case "1":
            plot()

        case "2":
            a_b()

        case "3":
            c_d()

        case "4":
            e()

        case "5":
            sys.exit(0)


if __name__ == '__main__':
    main_menu()
