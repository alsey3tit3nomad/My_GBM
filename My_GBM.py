import numpy as np
def My_MSE(Y, Y_hat):
    '''
    Parameters
    ----------
    Y : np.array
        array of target,
        \n shape: n x 1

    Y_hat : np.array
        array of outputs of model,
        \n shape: n x 1
    '''
    n = Y.shape[0]
    if (n == 0):
        return 0
    return 1/n * (Y - Y_hat).T @ (Y - Y_hat)


def sigmoid(t):
    return 1./(1 + np.exp(-t))

def My_Logloss(Y, Y_hat):
    eps = 1e-10
    return -np.mean(np.log(sigmoid(Y_hat) + eps) * Y + (1 - Y) * np.log(sigmoid(-Y_hat) + eps))

def Make_Gradient_by_input(loss_func, Y, Y_hat):
    if (loss_func is My_Logloss):
        p = sigmoid(Y_hat)
        return p - Y
    raise ValueError("Unsupported loss function")


class DecisionStump:
    def __init__(self, left_node=None, right_node=None, level=1, stop_min_el=1, loss_func=My_MSE, max_depth=10):
        # self.inf_value = inf_value
        self.left_node = left_node
        self.right_node = right_node
        self.right_value = None
        self.left_value = None
        self.feature_index = None
        self.threshold = None
        self.error = None
        self.level = level
        self.stop_min_el = stop_min_el
        self.is_leaf = False
        self.loss_func = loss_func
        self.max_depth=max_depth


    def fit(self, X, Y):
        '''
        Parameters
        ----------
        X : np.array
            array of objects : n x m

        Y : np.array
            array of outputs of model : n x 1
        '''
        if (X.shape[0] < self.stop_min_el or self.max_depth <= self.level):
            self.left_value = np.mean(Y)
            self.right_value = self.left_value
            self.error = self.loss_func(Y, np.full(Y.shape[0], self.left_value))
            self.is_leaf = True
            return self

        self.error = float('inf')
        n_feature = X.shape[1]
        for feature in range(n_feature):
            curr_threshold = np.unique(X[:, feature])
            curr_x = X[:, feature]          # Сделать нормальный критерий разбиения за линию/n log n
            for left_threshold in curr_threshold:
                left_mask = curr_x <= left_threshold
                right_mask = curr_x > left_threshold
                if np.any(left_mask) and np.any(right_mask):
                    left_mean = np.mean(Y[left_mask])
                    right_mean = np.mean(Y[right_mask])
                    preds = np.where(left_mask, left_mean, right_mean)
                    curr_error = self.loss_func(Y, preds)
                    if (curr_error < self.error):
                        self.left_value = left_mean
                        self.right_value = right_mean
                        self.error = curr_error
                        self.threshold = left_threshold
                        self.feature_index = feature
        if self.threshold is None or self.feature_index is None:
            self.left_value = np.mean(Y)
            self.right_value = self.left_value
            self.error = self.loss_func(Y, np.full(Y.shape[0], self.left_value))
            self.is_leaf = True
            return self
        self.left_node = DecisionStump(left_node=None, right_node=None, level=self.level+1, stop_min_el=self.stop_min_el, max_depth=self.max_depth)
        self.right_node = DecisionStump(left_node=None, right_node=None, level=self.level+1, stop_min_el=self.stop_min_el, max_depth=self.max_depth)
        self.left_node.fit(X[X[:, self.feature_index] <= self.threshold], Y[X[:, self.feature_index] <= self.threshold])
        self.right_node.fit(X[X[:, self.feature_index] > self.threshold], Y[X[:, self.feature_index] > self.threshold])

        return self

    def predict(self, X):
        if (self.is_leaf):
            return np.full(X.shape[0], int((self.left_value >= 0.5)))
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask
        preds = np.zeros(X.shape[0])
        preds[left_mask] = self.left_node.predict(X[left_mask])
        preds[right_mask] = self.right_node.predict(X[right_mask])

        return (preds >= 0.5).astype(int)

    def predict_proba(self, X):
        if (self.is_leaf):
            return np.full(X.shape[0], self.left_value)
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask
        preds = np.zeros(X.shape[0])
        preds[left_mask] = self.left_node.predict_proba(X[left_mask])
        preds[right_mask] = self.right_node.predict_proba(X[right_mask])

        return preds

class DecisionTree:
    def __init__(self, depth=1, min_elements=1, loss_func=My_MSE):
        self.depth=depth
        self.min_elements=min_elements
        self.loss_func = loss_func
        self.root = DecisionStump(loss_func=self.loss_func, stop_min_el=min_elements, max_depth=depth)

    def fit(self, X, Y):
        self.root.fit(X, Y)
        return self

    def predict(self, X):

        return self.root.predict(X)

    def predict_proba(self, X):
        return self.root.predict_proba(X)


class GradientBoostingMachineClassifier:
    def __init__(self, loss_func=My_Logloss, depth=3, learning_rate=0.1, min_elements=1, nums_of_models=10, initial_predict=None):
        self.loss_func = loss_func
        self.F = None
        self.models = []
        self.depth = depth
        self.learning_rate = learning_rate
        self.min_elements = min_elements
        self.nums_of_models=nums_of_models
        self.initial_prediction = initial_predict

    def fit(self, X, Y):
        prob = np.mean(Y)
        if (not self.initial_prediction):
            self.initial_prediction = np.log(prob/(1 - prob))
        self.F = np.ones(Y.shape[0]) * self.initial_prediction
        for _ in range(self.nums_of_models):
            S = -1 * Make_Gradient_by_input(self.loss_func, Y, self.F)
            curr_model = DecisionTree(self.depth, min_elements=self.min_elements, loss_func=My_MSE)
            curr_model.fit(X, S)
            updates = curr_model.predict_proba(X)
            self.F += self.learning_rate * updates
            self.models.append(curr_model)

        return self

    def predict_proba(self, X):
        curr_predict = self.initial_prediction * np.ones(X.shape[0])
        for model in self.models:
            curr_predict += self.learning_rate * model.predict_proba(X)
        return sigmoid(curr_predict)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=0, random_state=42)
    model = DecisionTree(depth=10, min_elements=10, loss_func=My_MSE)
    model.fit(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    print(f"Accuracy of DecisionTree on training data: {acc:.4f}")

    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=0, random_state=42)
    model = GradientBoostingMachineClassifier(loss_func=My_Logloss, depth=10)
    model.fit(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    print(f"Accuracy of My_GBM on training data: {acc:.4f}")