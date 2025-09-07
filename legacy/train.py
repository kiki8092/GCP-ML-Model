import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv('/data/train.csv')

df['gender']=pd.factorize(df['gender'])[0]
cleaned_df=df.dropna()
cleaned_df.shape[0]
x = cleaned_df.drop(["target","sno"], axis=1)
y = cleaned_df["target"]

np.random.seed(42)

log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Train the model
rs_log_reg.fit(x, y)


# Export the trained model
joblib.dump(rs_log_reg, 'trained_model.pkl')
