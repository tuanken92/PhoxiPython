import numpy as np


data = [{'x': 82.25, 'y': 296.25}, {'x': 82.25, 'y': 448.75}, {'x': 85.75, 'y': 451.25}, {'x': 87.5, 'y': 451.25}, {'x': 89.25, 'y': 450.0}, {'x': 96.25, 'y': 450.0}, {'x': 98.0, 'y': 448.75}, {'x': 285.25, 'y': 448.75}, {'x': 287.0, 'y': 450.0}, {'x': 295.75, 'y': 450.0}, {'x': 297.5, 'y': 448.75}, {'x': 341.25, 'y': 448.75}, {'x': 343.0, 'y': 450.0}, {'x': 383.25, 'y': 450.0}, {'x': 385.0, 'y': 451.25}, {'x': 400.75, 'y': 451.25}, {'x': 402.5, 'y': 450.0}, {'x': 414.75, 'y': 450.0}, 
{'x': 416.5, 'y': 448.75}, {'x': 418.25, 'y': 450.0}, {'x': 493.5, 'y': 450.0}, {'x': 495.25, 'y': 451.25}, {'x': 535.5, 'y': 451.25}, {'x': 537.25, 'y': 452.5}, {'x': 603.75, 'y': 452.5}, {'x': 605.5, 'y': 451.25}, {'x': 607.25, 'y': 451.25}, {'x': 609.0, 'y': 450.0}, {'x': 612.5, 'y': 450.0}, {'x': 619.5, 'y': 445.0}, {'x': 619.5, 'y': 296.25}, {'x': 451.5, 'y': 296.25}, {'x': 449.75, 'y': 297.5}, {'x': 427.0, 'y': 297.5}, {'x': 425.25, 'y': 296.25}, {'x': 421.75, 'y': 296.25}, {'x': 420.0, 'y': 297.5}, {'x': 409.5, 'y': 297.5}, {'x': 407.75, 'y': 298.75}, {'x': 397.25, 'y': 298.75}, {'x': 395.5, 'y': 297.5}, {'x': 392.0, 'y': 297.5}, {'x': 390.25, 'y': 296.25}, {'x': 225.75, 'y': 296.25}, {'x': 224.0, 'y': 297.5}, {'x': 222.25, 'y': 297.5}, {'x': 220.5, 'y': 298.75}, {'x': 211.75, 'y': 298.75}, {'x': 210.0, 'y': 297.5}, {'x': 204.75, 'y': 297.5}, {'x': 203.0, 'y': 296.25}, {'x': 194.25, 'y': 296.25}, {'x': 192.5, 'y': 297.5}, {'x': 182.0, 'y': 297.5}, {'x': 180.25, 'y': 298.75}, {'x': 161.0, 'y': 298.75}, {'x': 
159.25, 'y': 297.5}, {'x': 155.75, 'y': 297.5}, {'x': 154.0, 'y': 296.25}]

# Extract 'x' and 'y' values into separate lists
x_values = [point['x'] for point in data]
y_values = [point['y'] for point in data]

# Create NumPy arrays
x_array = np.array(x_values)
y_array = np.array(y_values)

# Combine 'x' and 'y' arrays into a single NumPy array if needed
combined_array = np.column_stack((x_array, y_array))

# Print the arrays
print("X Values:")
print(x_array)
print("\nY Values:")
print(y_array)


# Extract 'x' and 'y' values into separate lists
x_values = [point['x'] for point in data]
y_values = [point['y'] for point in data]

# Create NumPy arrays
x_array = np.array(x_values)
y_array = np.array(y_values)

# Combine 'x' and 'y' arrays into a single NumPy array if needed
combined_array = np.column_stack((x_array, y_array))

# Print the arrays
print("X Values:")
print(x_array)
print("\nY Values:")
print(y_array)


print("\ncombined_array:")
print(combined_array)

