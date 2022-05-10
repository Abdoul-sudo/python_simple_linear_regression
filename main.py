# SIMPLE LINEAR REGRESSION WITH PYTHON

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

y = np.array([68, 66, 68, 65, 69, 66, 68, 65, 71, 67, 68, 70])
x = np.array([65, 63, 67, 64, 68, 62, 70, 66, 68, 67, 69, 71])

plt.scatter(x, y) # Nuage de point

## Droite y = b1x + b0
linreg = LinearRegression()
x_resh = x.reshape(-1, 1)
linreg.fit(x_resh,y)
y_pred = linreg.predict(x_resh)
plt.plot(x_resh, y_pred, color='green')

# Equation of the line (y = mx + c) ou (y = b1*x + b0)
b1 = linreg.coef_
b0 = linreg.intercept_
print("Equation: y = "+str(b1)+"x + "+str(b0))

# --------

## Invertion des rôles de x et y (x = b_1y + b_0)
linreg2 = LinearRegression()
y_resh = y.reshape(-1, 1)
linreg2.fit(y_resh, x)
x_pred = linreg2.predict(y_resh)
plt.plot(y_resh, x_pred, color='purple')

# Equation of the inversed line (y = mx + c) ou (y = b_1*x + b_0)
b_1 = linreg2.coef_
b_0 = linreg2.intercept_
print("Equation: y_ = "+str(b_1)+"x + "+str(b_0))

# Intersection point
x_intersect = (b_0 - b0) / (b1 - b_1)
y_intersect = b1 * x_intersect + b0
print("Point d'intersection: I("+str(x_intersect)+", "+str(y_intersect)+")")

plt.plot(x_intersect,y_intersect,'ro') # Plot le point d'intersection r = red, o = dot

plt.show() # Affichage des plots 



