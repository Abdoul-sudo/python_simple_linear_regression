# SIMPLE LINEAR REGRESSION WITH PYTHON

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

y = np.array([68, 66, 68, 65, 69, 66, 68, 65, 71, 67, 68, 70])
x = np.array([65, 63, 67, 64, 68, 62, 70, 66, 68, 67, 69, 71])

linreg = LinearRegression()

x = x.reshape(-1, 1)

linreg.fit(x,y)

y_pred = linreg.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color='green')
plt.show()

# Equation of the line (y = mx + c) ou (y = b1*x + b0)
b1 = linreg.coef_
b0 = linreg.intercept_
print("Equation: y = "+str(b1)+"x + "+str(b0))


