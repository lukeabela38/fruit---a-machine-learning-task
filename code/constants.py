from enum import Enum
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
import xgboost as xgb

class Fruit(Enum):
    fruit_index: dict = {"apple": 1, "orange": 2, "lemon": 3}

class Data(Enum):
    variables: list = ["width", "weight", "height", "color_score"]

class TrainingAlgorithms(Enum):
    algorithms: list = [RandomForestClassifier, SVC, xgb.XGBClassifier, DecisionTreeClassifier]