```python
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
```


```python
# Supress warnings for clean notebook
import warnings
warnings.filterwarnings('ignore')
```


```python
dataset = pd.read_csv("Z:\Machine Learning\Google crash course on ML\Regularization\Melbourne_housing_FULL.csv")
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Suburb</th>
      <th>Address</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Price</th>
      <th>Method</th>
      <th>SellerG</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>...</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>CouncilArea</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Regionname</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abbotsford</td>
      <td>68 Studley St</td>
      <td>2</td>
      <td>h</td>
      <td>NaN</td>
      <td>SS</td>
      <td>Jellis</td>
      <td>3/09/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>126.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra City Council</td>
      <td>-37.8014</td>
      <td>144.9958</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abbotsford</td>
      <td>85 Turner St</td>
      <td>2</td>
      <td>h</td>
      <td>1480000.0</td>
      <td>S</td>
      <td>Biggin</td>
      <td>3/12/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>202.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra City Council</td>
      <td>-37.7996</td>
      <td>144.9984</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abbotsford</td>
      <td>25 Bloomburg St</td>
      <td>2</td>
      <td>h</td>
      <td>1035000.0</td>
      <td>S</td>
      <td>Biggin</td>
      <td>4/02/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>79.0</td>
      <td>1900.0</td>
      <td>Yarra City Council</td>
      <td>-37.8079</td>
      <td>144.9934</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Abbotsford</td>
      <td>18/659 Victoria St</td>
      <td>3</td>
      <td>u</td>
      <td>NaN</td>
      <td>VB</td>
      <td>Rounds</td>
      <td>4/02/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra City Council</td>
      <td>-37.8114</td>
      <td>145.0116</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Abbotsford</td>
      <td>5 Charles St</td>
      <td>3</td>
      <td>h</td>
      <td>1465000.0</td>
      <td>SP</td>
      <td>Biggin</td>
      <td>4/03/2017</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>150.0</td>
      <td>1900.0</td>
      <td>Yarra City Council</td>
      <td>-37.8093</td>
      <td>144.9944</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
dataset.nunique()
```




    Suburb             351
    Address          34009
    Rooms               12
    Type                 3
    Price             2871
    Method               9
    SellerG            388
    Date                78
    Distance           215
    Postcode           211
    Bedroom2            15
    Bathroom            11
    Car                 15
    Landsize          1684
    BuildingArea       740
    YearBuilt          160
    CouncilArea         33
    Lattitude        13402
    Longtitude       14524
    Regionname           8
    Propertycount      342
    dtype: int64




```python
dataset.shape
```




    (34857, 21)




```python
cols_to_use = ['Suburb','Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea','Price']
dataset = dataset[cols_to_use]
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Suburb</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Method</th>
      <th>SellerG</th>
      <th>Regionname</th>
      <th>Propertycount</th>
      <th>Distance</th>
      <th>CouncilArea</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>SS</td>
      <td>Jellis</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>126.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>S</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>202.0</td>
      <td>NaN</td>
      <td>1480000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>S</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>79.0</td>
      <td>1035000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Abbotsford</td>
      <td>3</td>
      <td>u</td>
      <td>VB</td>
      <td>Rounds</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Abbotsford</td>
      <td>3</td>
      <td>h</td>
      <td>SP</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>150.0</td>
      <td>1465000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.shape
```




    (34857, 15)




```python
dataset.isna().sum()
```




    Suburb               0
    Rooms                0
    Type                 0
    Method               0
    SellerG              0
    Regionname           3
    Propertycount        3
    Distance             1
    CouncilArea          3
    Bedroom2          8217
    Bathroom          8226
    Car               8728
    Landsize         11810
    BuildingArea     21115
    Price             7610
    dtype: int64




```python
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)
dataset.isna().sum()
```




    Suburb               0
    Rooms                0
    Type                 0
    Method               0
    SellerG              0
    Regionname           3
    Propertycount        0
    Distance             0
    CouncilArea          3
    Bedroom2             0
    Bathroom             0
    Car                  0
    Landsize         11810
    BuildingArea     21115
    Price             7610
    dtype: int64




```python
dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())

dataset.isna().sum()
```




    Suburb              0
    Rooms               0
    Type                0
    Method              0
    SellerG             0
    Regionname          3
    Propertycount       0
    Distance            0
    CouncilArea         3
    Bedroom2            0
    Bathroom            0
    Car                 0
    Landsize            0
    BuildingArea        0
    Price            7610
    dtype: int64




```python
dataset.dropna(inplace=True)
dataset.isna().sum()
```




    Suburb           0
    Rooms            0
    Type             0
    Method           0
    SellerG          0
    Regionname       0
    Propertycount    0
    Distance         0
    CouncilArea      0
    Bedroom2         0
    Bathroom         0
    Car              0
    Landsize         0
    BuildingArea     0
    Price            0
    dtype: int64




```python
dataset.shape
```




    (27244, 15)




```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Suburb</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Method</th>
      <th>SellerG</th>
      <th>Regionname</th>
      <th>Propertycount</th>
      <th>Distance</th>
      <th>CouncilArea</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>S</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>202.0</td>
      <td>160.2564</td>
      <td>1480000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>S</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>79.0000</td>
      <td>1035000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Abbotsford</td>
      <td>3</td>
      <td>h</td>
      <td>SP</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>150.0000</td>
      <td>1465000.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Abbotsford</td>
      <td>3</td>
      <td>h</td>
      <td>PI</td>
      <td>Biggin</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>94.0</td>
      <td>160.2564</td>
      <td>850000.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Abbotsford</td>
      <td>4</td>
      <td>h</td>
      <td>VB</td>
      <td>Nelson</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>Yarra City Council</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>142.0000</td>
      <td>1600000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset = pd.get_dummies(dataset, drop_first=True)
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rooms</th>
      <th>Propertycount</th>
      <th>Distance</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>Price</th>
      <th>Suburb_Aberfeldie</th>
      <th>...</th>
      <th>CouncilArea_Moorabool Shire Council</th>
      <th>CouncilArea_Moreland City Council</th>
      <th>CouncilArea_Nillumbik Shire Council</th>
      <th>CouncilArea_Port Phillip City Council</th>
      <th>CouncilArea_Stonnington City Council</th>
      <th>CouncilArea_Whitehorse City Council</th>
      <th>CouncilArea_Whittlesea City Council</th>
      <th>CouncilArea_Wyndham City Council</th>
      <th>CouncilArea_Yarra City Council</th>
      <th>CouncilArea_Yarra Ranges Shire Council</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>202.0</td>
      <td>160.2564</td>
      <td>1480000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>79.0000</td>
      <td>1035000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>150.0000</td>
      <td>1465000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>94.0</td>
      <td>160.2564</td>
      <td>850000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>4019.0</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>142.0000</td>
      <td>1600000.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 745 columns</p>
</div>




```python

```
