import numpy as np
import pandas as pd
from preprocessing import set_missing_ages, set_cabin_type, one_hot_encoder, standard_scaler
from logistic_regression import logistic_regression, predict, adam, sgd, adagrad, rmsprop

if __name__ == '__main__':
    data_train = pd.read_csv('./titanic/train.csv')

    set_cabin_type(data_train)
    rfr = set_missing_ages(data_train)
    data_train = one_hot_encoder(data_train, ['Embarked', 'Cabin', 'Sex', 'Pclass'])
    standard_scaler(data_train, ['Age', 'Fare', 'Parch', 'SibSp'])

    train_df = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values
    x = train_np[:, 1:]
    y = train_np[:, 0]
    # weights = adam(x, y, 0.01, 10000, 0.9, 0.999, 1e-8)
    # weights = logistic_regression(x, y, 0.01, 10000)
    # weights = sgd(x, y, 0.01, 10000)
    # weights = adagrad(x, y, 0.01, 10000, 1e-8)
    weights = rmsprop(x, y, 0.01, 10000, 0.9, 1e-8)
    data_test = pd.read_csv('./titanic/test.csv')
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].values
    x = null_age[:, 1:]
    predicted_ages = rfr.predict(x)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predicted_ages
    set_cabin_type(data_test)
    data_test = one_hot_encoder(data_test, ['Embarked', 'Cabin', 'Sex', 'Pclass'])

    test_df = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    test_np = test_df.values
    test_np = np.hstack((np.ones((test_np.shape[0], 1)), test_np))
    x_test = test_np[:, :]
    survived_predict = predict(x_test,  weights)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'], 'Survived': survived_predict.reshape(-1).astype(np.int32)})
    result.to_csv("predicted_result.csv", index=False)

