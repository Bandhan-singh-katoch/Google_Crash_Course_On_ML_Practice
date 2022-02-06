```python
import numpy as np
import pandas as pd
```


```python
# creating 5x2 Numpy array
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_col_names = ['temperature','activity']

# create a dataframe
my_dataframe = pd.DataFrame(data=my_data,columns=my_col_names)

print(my_dataframe)
```

       temperature  activity
    0            0         3
    1           10         7
    2           20         9
    3           30        14
    4           40        15
    

Adding a new column to a DataFrame


```python
my_dataframe['adjusted'] = my_dataframe['activity']+2
print(my_dataframe)
```

       temperature  activity  adjusted
    0            0         3         5
    1           10         7         9
    2           20         9        11
    3           30        14        16
    4           40        15        17
    


```python
print("Rows #0, #1, and #2:")
print(my_dataframe.head(3),"\n")

print("Row #3")
print(my_dataframe.iloc[[3]],"\n")

print("Row #1, #2, #3")
print(my_dataframe[1:4],"\n")

print("temperature column")
print(my_dataframe["temperature"])
```

    Rows #0, #1, and #2:
       temperature  activity  adjusted
    0            0         3         5
    1           10         7         9
    2           20         9        11 
    
    Row #3
       temperature  activity  adjusted
    3           30        14        16 
    
    Row #1, #2, #3
       temperature  activity  adjusted
    1           10         7         9
    2           20         9        11
    3           30        14        16 
    
    temperature column
    0     0
    1    10
    2    20
    3    30
    4    40
    Name: temperature, dtype: int32
    

Task 1: Create a DataFrame

Do the following:

1.Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named Eleanor, Chidi, Tahani, and Jason. Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.

2.Output the following:

-> the entire DataFrame
-> the value in the cell of row #1 of the Eleanor column

3.Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason.

To complete this task, it helps to know the NumPy basics covered in the NumPy UltraQuick Tutorial.


```python
col_name = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

my_new_data = np.random.randint(0,100, size=(3,4))

my_data_frame = pd.DataFrame(data=my_new_data,columns=col_name)

#the entire DataFrame
print(my_data_frame,"\n")

#the value in the cell of row #1 of the Eleanor column
print("the value in the cell of row #1 of the Eleanor column",my_data_frame['Eleanor'][0],"\n")

# Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason.
my_data_frame['Janet'] = my_data_frame['Tahani']+my_data_frame['Jason']
print(my_data_frame)
```

       Eleanor  Chidi  Tahani  Jason
    0       51     26      34     75
    1       84     15      78     82
    2       29     29      96     77 
    
    the value in the cell of row #1 of the Eleanor column 51 
    
       Eleanor  Chidi  Tahani  Jason  Janet
    0       51     26      34     75    109
    1       84     15      78     82    160
    2       29     29      96     77    173
    
