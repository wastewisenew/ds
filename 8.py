import numpy as np
class Node:
def __init__(self, feature=None, value=None, results=None, true_branch=None,
false_branch=None):
self.feature = feature # Feature to split on
self.value = value # Value of the feature
self.results = results # None for nodes, holds value for leaf nodes
self.true_branch = true_branch # Subtree for when the condition is true
self.false_branch = false_branch # Subtree for when the condition is false
def unique_vals(rows, col):
return set([row[col] for row in rows])
def class_counts(rows):
counts = {}
# A dictionary of label -> count.
for row in rows:
# In our dataset format, the label is always the last column
label = row[-1]
if label not in counts:
counts[label] = 0
counts[label] += 1
return counts
def is_numeric(value):
return isinstance(value, int) or isinstance(value, float)
def gini(rows):
counts = class_counts(rows)
impurity = 1
for lbl in counts:
prob_of_lbl = counts[lbl] / float(len(rows))
impurity -= prob_of_lbl**2
return impurity
def info_gain(left, right, current_uncertainty):
p = float(len(left)) / (len(left) + len(right))
return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
def find_best_split(rows):
best_gain = 0
best_feature = None
current_uncertainty = gini(rows)
n_features = len(rows[0]) - 1
for col in range(n_features): # For each feature
values = set([row[col] for row in rows]) # Unique values in the column
for val in values: # For each value
true_rows, false_rows = split(rows, col, val)
if len(true_rows) == 0 or len(false_rows) == 0:
continue
gain = info_gain(true_rows, false_rows, current_uncertainty)
if gain >= best_gain:
best_gain, best_feature = gain, (col, val)
return best_gain, best_feature
def split(rows, col, value):
true_rows, false_rows = [], []
for row in rows:
if row[col] == value:
true_rows.append(row)
else:
false_rows.append(row)
return true_rows, false_rows
def build_tree(rows):
gain, (feature, value) = find_best_split(rows)
if gain == 0:
return Node(results=class_counts(rows))
true_rows, false_rows = split(rows, feature, value)
true_branch = build_tree(true_rows)
false_branch = build_tree(false_rows)
return Node(feature, value, true_branch=true_branch, false_branch=false_branch)
def print_tree(node, spacing=""):
if node.results is not None: # Leaf node
print(spacing + "Predict", node.results)
return
print(spacing + str(node.feature) + ':' + str(node.value))
print(spacing + '--> True:')
print_tree(node.true_branch, spacing + " ")
print(spacing + '--> False:')
print_tree(node.false_branch, spacing + " ")
def classify(row, node):
if node.results is not None:
return node.results
if is_numeric(row[node.feature]):
if row[node.feature] >= node.value:
return classify(row, node.true_branch)
else:
return classify(row, node.false_branch)
else:
if row[node.feature] == node.value:
return classify(row, node.true_branch)
else:
return classify(row, node.false_branch)
dataset = [
['Low', 'Low', 2, 'No', 'Yes'],
['Low', 'Med', 4, 'Yes', 'Yes'],
['Low', 'Low', 4, 'No', 'Yes'],
['Low', 'Med', 4, 'No', 'No'],
['Low', 'High', 4, 'No', 'No'],
['Med', 'Med', 4, 'No', 'No'],
['Med', 'Med', 4, 'Yes', 'Yes'],
['Med', 'High', 2, 'Yes', 'No'],
['Med', 'High', 5, 'No', 'Yes'],
['High', 'Med', 4, 'Yes', 'Yes'],
['High', 'Med', 2, 'Yes', 'Yes'],
['High', 'High', 2, 'Yes', 'No'],
['High', 'High', 5, 'Yes', 'Yes']
]
# Build the tree
tree = build_tree(dataset)
# Print the tree
print("Decision Tree:")
print_tree(tree)
# Test the tree
test_data = ['Med', 'Low', 4, 'No'] # Test data instance
print("\nClassifying Test Data:")
print("Test Data:", test_data)
print("Classification:", classify(test_data, tree))