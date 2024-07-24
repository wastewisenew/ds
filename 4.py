import pandas as pd
import matplotlib.pyplot as plt
mtcars = pd.read_csv('desktop/mtcars.csv')
plt.hist(mtcars['mpg'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Miles per gallon')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Miles per Gallon')
plt.show()