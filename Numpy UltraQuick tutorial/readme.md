```python
import numpy as np
```


```python
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)
```

    [1.2 2.4 3.5 4.7 6.1 7.2 8.3 9.5]
    


```python
two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)
```

    [[ 6  5]
     [11  7]
     [ 4  8]]
    


```python
zero_array = np.zeros(5)  # np.ones(5)
print(zero_array)
```

    [0. 0. 0. 0. 0.]
    


```python
sequence_of_integers = np.arange(4,13)
print(sequence_of_integers)
```

    [ 4  5  6  7  8  9 10 11 12]
    


```python
# Populate arrays with random numbers
random_numb_betw_50and100 = np.random.randint(50,100,5)
print(random_numb_betw_50and100)
```

    [93 54 69 68 61]
    


```python
random_floats_between_0_and_1 = np.random.random([5])
print(random_floats_between_0_and_1)
```

    [0.77754863 0.71441465 0.81843238 0.9153006  0.89756894]
    


```python
random_float_betw_2and3 = random_floats_between_0_and_1 + 2.0
print(random_float_betw_2and3)
```

    [2.77754863 2.71441465 2.81843238 2.9153006  2.89756894]
    


```python
random_numb_betw_150and300 = random_numb_betw_50and100 * 3
print(random_numb_betw_150and300)
```

    [279 162 207 204 183]
    

Task 1: Create a Linear Dataset

Your goal is to create a simple dataset consisting of a single feature and a label as follows:

Assign a sequence of integers from 6 to 20 (inclusive) to a NumPy array named feature.

Assign 15 values to a NumPy array named label such that:
 label = (3)(feature) + 4
For example, the first value for label should be:
label = (3)(6) + 4 = 22

```python
feature = np.arange(6,21)
print(feature)
label = feature*3 + 4
print(label)
```

    [ 6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    [22 25 28 31 34 37 40 43 46 49 52 55 58 61 64]
    

Task 2: Add Some Noise to the Dataset

To make your dataset a little more realistic, insert a little random noise into each element of the label array you already created. To be more precise, modify each value assigned to label by adding a different random floating-point value between -2 and +2.

Don't rely on broadcasting. Instead, create a noise array having the same dimension as label.


```python
noise = np.random.randint(-2,2,15)
print(noise)
label = noise + label
print(label)
```

    [ 1 -2 -1  0  0 -2 -2 -1 -1  1 -2  1 -1 -2 -1]
    [23 23 27 31 34 35 38 42 45 50 50 56 57 59 63]
    


```python

```
