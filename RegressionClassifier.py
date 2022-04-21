from sklearn.metrics import accuracy_score


class Classification:

    def __init__(self, model):
        self.reg = model

    def fit(self, X_train, y_train):
        """
        Fit the  regression model on the training data
        :param X_train: The training set used to classify the response variable
        :param y_train: The response variable
        """
        self.reg.fit(X_train, y_train.ravel())

    def predict(self, X_test):
        """
        Predict the response variable using a test set
        :param X_test: Test variables used to predict the response variable
        :return: A 1D array that contains the predicted variables
        """
        predictions = self.reg.predict(X_test).tolist()
        predictions = [[prediction] for prediction in predictions]
        return predictions

    def acc(self, predicted, y_test):
        """
        Calculate the accuracy of the fitted model
        :param predicted: The predicted response variable
        :param y_test: The labels of the predicted variables to check their correctness
        :return: Accuracy of the model in percentage
        """

        return accuracy_score(y_test, predicted) * 100