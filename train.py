import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def q2(data):
    plt.figure()
    plt.plot(data['year'], data['days'])
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig("data_plot.jpg")

def q3(data):
    year_data = data['year']

    min_year = year_data.min()
    max_year = year_data.max()

    normalized = (year_data - min_year) / (max_year - min_year)
    X_normalized = np.column_stack((normalized, np.ones(len(data))))

    print("Q3:")
    print(X_normalized)

    return X_normalized

def q4(data, X_normalized):
    y = data['days'].values
    X_transpose = X_normalized.T

    inverse = np.linalg.inv(X_transpose @ X_normalized)
    weights = inverse @ X_transpose @ y

    print("Q4:")
    print(weights)

    return weights

def q5(data, X_normalized, learning_rate, iterations):
    n = len(data['days'].values)
    y = data['days'].values
    w = np.zeros(2)
    losses = []

    for t in range(iterations):
        if t % 10 == 0:
            print(w)
        y_pred = X_normalized @ w
        dw = (X_normalized.T @ (y_pred - y)) / n
        w -= learning_rate * dw
        
        loss = np.sum((y_pred - y) ** 2) / (2 * n)
        losses.append(loss)
    return losses, w

def q6(data, w):
    year_data = data['year']

    min_year = year_data.min()
    max_year = year_data.max()

    x_normalized = (2023 - min_year) / (max_year - min_year)
    y_hat = w[0] * x_normalized + w[1]

    print("Q6: " + str(y_hat))

def q7(weights):
    w = weights[0]
    if w > 0:
        print("Q7a: >")
    elif w < 0:
        print("Q7a: <")
    else:
        print("Q7a: =")
    
    print("Q7b Case 1: w>0\n - The number of ice days increases over time.\nCase 2: w<0\n - The number of ice days decreases over time.\nCase 3: w=0\n - There is no relation ")
    
def q8(data, weights):
    year_data = data['year']

    min_year = year_data.min()
    max_year = year_data.max()

    x_star = min_year + (max_year - min_year) * (-(weights[1]) / weights[0])

    print("Q8a: " + str(x_star))
    print("Q8b: This prediction is a good estimation but it is not 100% accurate due to many limitig factors. Some of these factors incude assuming linearity of complex natural phenomenon and basing this on past data.")    

def runHW5():
    if len(sys.argv) != 4:
        print("Incorrect argument count")
        sys.exit(1)
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])

    data = pd.read_csv(filename)

    #Question 2
    q2(data)

    #Question 3
    X_normalized = q3(data)

    #Question 4
    weights = q4(data, X_normalized)

    #Question 5
    print("Q5a:")
    losses, w  = q5(data, X_normalized, learning_rate, iterations )
    print("Q5b: " + str(learning_rate))
    print("Q5c: " + str(iterations))
    #Question 5 - saving loss plot
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss_plot.jpg")
    
    #Question 6
    q6(data, weights)

    #Question 7
    q7(weights)

    #Question 8
    q8(data, weights)

if __name__ == "__main__":
    runHW5()