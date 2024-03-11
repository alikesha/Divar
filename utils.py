
import time
import re



from sklearn.model_selection import train_test_split , cross_val_score, KFold ,GridSearchCV , RandomizedSearchCV

from sklearn.linear_model import LinearRegression , Ridge

from sklearn.metrics import r2_score , mean_squared_error, mean_absolute_error






def parameter_finder (model, parameters):

    start = time.time()

    grid = GridSearchCV(model,
                        param_grid = parameters,
                        refit = True,
                        cv = KFold(shuffle = True, random_state = 1),
                        n_jobs = -1)
    grid_fit = grid.fit(x_train, y_train)
    y_train_pred = grid_fit.predict(x_train)
    y_pred = grid_fit.predict(x_test)

    train_score =grid_fit.score(x_train, y_train)
    test_score = grid_fit.score(x_test, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    model_name = str(model).split('(')[0]

    end = time.time()

    print(f"The best parameters for {model_name} model is: {grid_fit.best_params_}")
    print("--" * 10)
    print(f"(R2 score) in the training set is {train_score:0.2%} for {model_name} model.")
    print(f"(R2 score) in the testing set is {test_score:0.2%} for {model_name} model.")
    print(f"RMSE is {RMSE:,} for {model_name} model.")
    print("--" * 10)
    print(f"Runtime of the program is: {end - start:0.2f}")


    return train_score, test_score, RMSE