import statistics as st

# Given dataset
x = [115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4, 90.4]

# Calculate mean
sum_x = sum(x)
mean = sum_x / len(x)

# Sort the dataset in ascending order
x.sort()

# Calculate median
if len(x) % 2 == 0:
    a = x[len(x)//2]
    b = x[(len(x)//2) - 1]
    median = (a + b) / 2
else:
    median = x[len(x)//2]

# Calculate mode
mx = 0
mode = 0
for i in x:
    if x.count(i) >= mx:
        mx = x.count(i)
        mode = i

# Calculate variance
var = sum((i - mean)**2 for i in x) / (len(x) - 1)

# Calculate standard deviation
std = var**0.5

# Find minimum and maximum values
xm = min(x)
xmax = max(x)

# Min-Max scaling
MMS = [(i - xm) / (xmax - xm) for i in x]

# Standardization (Z-score normalization)
SS = [(i - mean) / std for i in x]

# Print results
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Standard Deviation:", std)
print("Variance:", var)
print("Min-Max Scaling:", MMS)
print("Standardization (Z-score):", SS)
