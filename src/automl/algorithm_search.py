"""This is a script that will be used to search for the best regression algorithm/model for the given dataset (or as a whole)."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

### simple function to just return a model beffore developing the search method
def return_model():
    return RandomForestRegressor()
