# Import necessary libraries
import pandas as pd
from matplotlib import pyplot

# Read the CSV file into a DataFrame
data = pd.read_csv('/Users/aaronmackenzie/Desktop/Programming/Food-Truck-LineReg.csv')

# Calculate mean of X and Y
Xmean = data['X'].mean()
Ymean = data['Y'].mean()

# Calculate XY mean
data['XY'] = data['X'] * data['Y']
XYmean = data['XY'].mean()

# Calculate X^2 mean
data['Xp2'] = data['X']**2
Xp2mean = data['Xp2'].mean()

# Calculate the slope (m) and y-intercept (c) of the regression line
m = ((XYmean) - (Xmean * Ymean)) / (Xp2mean - (Xmean**2))
c = Ymean - (m * Xmean)

# Predicted values using the regression line
data['ypred'] = (m * data['X'] + c)

# Scatter plot of the data points and the regression line
pyplot.scatter(data['X'], data['Y'])
pyplot.plot(data['X'], data['ypred'])

# Calculate error, SSE (Sum of Squares Error)
data['error2'] = (data['Y'] - data['ypred'])**2
sse = data['error2'].sum()

# Calculate SST (Total Sum of Squares)
data['y-ymean2'] = (data['Y'] - Ymean)**2
sst = data['y-ymean2'].sum()

# Calculate SSR (Sum of Squares Regression)
data['ypred-ymean2'] = (data['ypred'] - Ymean)**2
ssr = data['ypred-ymean2'].sum()

# Calculate R-squared (coefficient of determination)
r2 = ssr / sst

# Print the calculated statistics
print("SSE =", sse)
print("SST =", sst)
print("SSR =", ssr)
print("R-squared =", r2)
