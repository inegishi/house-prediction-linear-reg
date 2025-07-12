import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def convert_bin(col):
    if col == "yes" or col == "furnished":
        return 1
    elif col == "semi-furnished":
        return 0.5
    return 0

def minimize(col):
    return col/1000 if col <= 10000 else col/ 1000000

#'price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad','guestroom', 'basement', 'hotwaterheating', 'airconditioning','parking', 'prefarea', 'furnishingstatus'
#every row is filled

def compute_cost(x,y,w,b):
    m = x.shape[0]
    total = 0
    for i in range(m):
        fw_b = np.dot(x[i],w) + b
        total += (fw_b - y[i])**2

    return total * 1/(2*m)

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw_sum, dj_db_sum  = 0, 0
    for i in range(m):
        fw_b = np.dot(w,x[i]) + b
        dj_dw_sum += (fw_b - y[i]) * x[i]
        dj_db_sum += fw_b - y[i]
    dj_db_sum = dj_db_sum/m
    dj_dw_sum = dj_dw_sum/m
    return dj_db_sum , dj_dw_sum

def gradient_descent(x, y, w, b, alpha, num_iters): 

    cost_hist = []
    for i in range(num_iters):
        cost = compute_cost(x,y,w,b)
        cost_hist.append(cost)
        dj_db,dj_dw = compute_gradient(x,y,w,b)
        if i%500 == 0:
            print(f"Cost: {cost}")
        w = w - alpha * dj_dw
        b = b - alpha * dj_db


    return w,b

def prediction(w,x,b):
    p = np.dot(x,w) + b
    return p    


df = pd.read_csv("Housing.csv")


feature_df = df.drop(columns=["price"])
price_df = df["price"]
cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea", "furnishingstatus"]
for col in cols:
    feature_df[col] = feature_df[col].apply(convert_bin)

feature_df["area"] = feature_df["area"].apply(minimize)
price_df= price_df.apply(minimize)
# print(feature_df["area"].head)
# print(price_df.head)

x_train, x_test, y_train, y_test = train_test_split(feature_df, price_df, test_size=0.2, shuffle=False)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()



#TRAINNG

w_init = np.zeros(x_train.shape[1])
b_init = 0

w,b = gradient_descent(x_train,y_train,w_init,b_init,0.001,10000)
print(f"final {w,b}")   

wfinal= np.array([0.12981266, 0.14758947, 1.01338298, 0.38360924, 0.66565378,
       0.25603033, 0.23721181, 0.86217169, 0.79916789, 0.34778573,
       0.69028301, 0.27806497])
bfinal = 0.3646216928316859

# pred = prediction(wfinal,x_test,bfinal)
# for i in range(len(pred)):
#     print(f"prediction: {pred[i]}, actual {y_test[i]}, percent error: {abs((pred[i]-y_test[i])/y_test[i])*100}")