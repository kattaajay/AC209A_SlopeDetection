---
title: compare roll angle amplitude
notebook: Report-IntroductionandEDA.ipynb
nav_include:2
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.api import OLS
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
import itertools;
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from itertools import product
from collections import OrderedDict   
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
from matplotlib import colors
import seaborn as sns

%matplotlib inline

sns.set_context("poster")
```


    /Users/kate_zym/anaconda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools



### Experiment Details:

The sloped walking protocol involved 6 subjects walking at 7 different conditions. The walking slope is collected at discrete slopes of -10%, -5%, 0%, 5%, 10%, 15%, and 20% set by the treadmill. And the suit data was collected at various speeds of 1 m/s, 1.5 m/s, ..., 3.5 m/s, 4 m/s.

!!!!!!!!!!!!!! EXPLAIN MORE IF TIME

<img src="experiment.png"  style="width: 500px;">
<center> Fig 1. Experiment set-up <center>

### Description of Data:

The Hip-Only soft exoskeleton has 3
IMUs (MTi-3 AHRS, Xsens Technologies B.V., Enschede, Netherlands) on the abdomen and the
anterior part of each thigh. Each IMU can measure 3-axes linear and angular
motion using built in accelerometer and gyroscopes. The outputs of the IMU are relevant to the suit because it gives information about the phase of user's gait cycle which determines when assistance should be provided. An image of the particular IMU we are using is shown below.

<img src="IMU.jpg"  style="width: 800px;">
<center> Fig 2. Image of Xsens IMU from Xsens User Manual <center>


During the sloped walking experiments on the treadmill, only certain IMU outputs are selected for streaming. The roll, pitch and yaw angles, angular velocity and linear accelerations were collected. Note the yaw angle is interpreted as heading and measured in teh East-North-Up coordinate system, whereas the roll and pitch angles are measured in the local coordinate frame. The sequence of rotation for Euler angles uses the aerospace convention for rotation from global frame to location frame. Referencing the local coordinate frame in Fig. 1, the IMU sensor is worn such that *x* axis is normal to the sagital plane and points medially, *y* axis is normal to the coronal plane and points anteriorially, and *z* axis is normal to the transverse plane and points vertically down.

<img src="anatomical_planes.png"  style="width: 500px;">
<center> Fig 3. Image of Xsens IMU from Xsens User Manual <center>

!!!!!!!!!!!!!! *ADD MORE LATER IF THERES TIME*


The suit has an algorithm that segments gait cycles using the maximum hip flexion angle, and the percent gait cycle estimates is another data column. There is also a flag to indicate whether the subject is running determined from the acceleration readings from the abdomen IMU; it is numeric rather than binary because it encodes several different flags together.

From the raw system data, we divided the data into strides. We chose to use each stride as
an independent observation. For the IMU data, we divided each angle, angular velocity and
linear acceleration into 50 predictors at every 2% GC. (For instance, ‘Angle X at 24%’ gait cycle
is a separate predictor than ‘Angle X at 26%’ gait cycle.) We also found the maximums and minimums of each angle, angular velocity and linear acceleration, and used them as additional predictors. Finally, the mean of each IMU output are also included as
additional predictors.

In addition to IMU data, we included the subject number, height and weight for each
observation. Subject information are very important for evaluating wearable devices as walking
kinematics may be slightly different for each person. Finally, sloped walking data was also
collected at different walking speeds (1 m/s, 1.5 m/s, ..., 3.5 m/s, 4 m/s), thus walking speed is
another important predictor.

For the outcome variable, the walking slope is collected at discrete slopes of -10%, -5%, 0%,
5%, 10%, 15%, and 20%.

*Note: we used MATLAB to perform data cleaning and to exact and assemble all predictors
mentioned above. We have a total of 1820 predictors and 5006 observations.

### Additional Feature Engineering

After some initial data exploration,  we noticed that the kinematic data seem shifted per subject or testing condition. This is likely due to the inconsistent placement of the wearable sensors during each donning. To compensate for these shifts in data range, we decided to add more predictors of each IMU signal subtracting the mean of that signal during each stride.

We used the orignal column names and added "_mm" (which stand for *minus mean*) for each of these newly engineered predictors. The same subtraction of mean was done to the *max* and *min* values of each predictor.

With these engineered features, we have 3251 predictors in total.

### Further Data Manipulation - Removing predictors

In this third stage of manipuating our data, we decided to remove predictors that are difficult to collect or unreliable during real time data collection of the suit. In order to better minic a real-time slope estimation algorithm, the *angle Z* (or yaw angle), *acceleration x*, and *accleration y* are removed.

Moreover, we decided not to use the *reference slope* as a predictor because of the systems lack of ability to estimate walking speed during overground walking.

Finally, *percentage gait cycle* and *running flag* predictors were removed because of their irrelevance to slope estimation.

After removing these unncessary predictors, we have a total of 1928 predictors.

### Standardization

Because most of our data physical kinematic values, no standardization was required as a natural boundary of reasonable values have been set by body geometry. However, for the need of unbiased regularization and PCA, we standardized the data when using those methods

-------------------


### EDA for 3-Class Classification (flatground, uphill, downhill)

Methods for Initial Data Exploration:
First for our classification problem, we split the data into three sets - flat ground, uphill, and downhill, denoted by class 0, class 1 and class -1 respectively. We plotted the IMU data of each gait cycle (50 features per gait cycle), and compared the three groups against each other.



```python
### 1. READING DATA
df = pd.read_csv('Data/dataAll_csv8_withMM.csv')

df["slope_class"] = 0
df["slope_class"][df['ref_slope'] < 0] = -1
df["slope_class"][df['ref_slope'] == 0]= 0
df["slope_class"][df['ref_slope'] > 0]= 1
```


    /Users/kate_zym/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    /Users/kate_zym/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    /Users/kate_zym/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      import sys




```python
def standardize(x, x_ref):
    # return standardized data
    mean = np.mean(x_ref);
    std = np.std(x_ref);  

    x_stand = (x - mean)/std;
    return x_stand
```




```python
df_downhill = df[df['ref_slope'] < 0]
df_flat = df[df['ref_slope'] == 0]
df_uphill = df[df['ref_slope'] > 0]

msk1 = np.random.rand(len(df_downhill)) < 1000 /len(df_downhill)
msk2 = np.random.rand(len(df_flat)) < 1000 /len(df_flat)
msk3 = np.random.rand(len(df_uphill)) < 1000 /len(df_uphill)

df_downhill = df_downhill[msk1]
df_flat = df_flat[msk2]
df_uphill = df_uphill[msk3]


downhill_stand = standardize(df_downhill, df_downhill)
flat_stand = standardize(df_flat, df_flat)
uphill_stand = standardize(df_uphill, df_uphill)


df.shape, df_downhill.shape, df_flat.shape, df_uphill.shape, downhill_stand.shape
```





    ((5006, 3252), (1009, 3252), (859, 3252), (992, 3252), (1009, 3252))



Some kinematic variables showed more promise for separating the different slope conditions.
Some interesting findings are shown below:



```python
sns.set_context("poster")

a = 52

for i in range(100):
    plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[8*i, a : a+51]), 'r', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[8*i, a : a+51]), 'b', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[8*i, a : a+51]), 'g', alpha = 0.1)
    

plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[i+1, a : a+51]), 'r', alpha = 0.5, label = 'downhill')
plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[i+1, a : a+51]), 'b', alpha = 0.5, label = 'flat')
plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[i+1, a : a+51]), 'g', alpha = 0.5, label = 'uphill')
    
plt.legend()
plt.title("Right Thigh Extension Angle (" + str(df.columns[a][:-2] + ")"))
plt.xlabel("Percentage of Gait Cycle [%]")
plt.ylabel("Angle [deg]")
```





    <matplotlib.text.Text at 0x1432a24a8>




![png](Report-IntroductionandEDA_files/Report-IntroductionandEDA_20_1.png)


#### Comments:
* Plotting the thigh extension angle during one gait cycle shows promising differences between the three classes. Around 40% gait cycle, X-axis angle seems separate downhill, flat, and uphill condition quite well.



```python
plt.plot(df_downhill['ref_speed'], df_downhill['angleX_R_40'],'s', c = 'r', alpha = 0.3,label = 'downhill')
plt.plot(df_flat['ref_speed'], df_flat['angleX_R_40'], 's', c='g', alpha = 0.3, label = 'flat')
plt.plot(df_uphill['ref_speed'], df_uphill['angleX_R_40'], 's', c='b', alpha = 0.3, label = 'uphill')

plt.legend()
plt.ylabel('Thigh Angle at 40% GC [deg]')
plt.xlabel('Reference Speed [m/s]')
```





    <matplotlib.text.Text at 0x146fed400>




![png](Report-IntroductionandEDA_files/Report-IntroductionandEDA_22_1.png)


#### Comments:
* Plotting the Thigh angle at 40% gait cycle versus the reference speed shows each slope class is clustered at different regions
* Although having reference speed as a predictor will greatly help the classification performance, the reference speed cannot be used as overground walking contains time-varying speeds that cannot be estimated accurately




```python
a = 52+54

for i in range(100):
    plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[8*i, a : a+51]), 'r', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[8*i, a : a+51]), 'b', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[8*i, a : a+51]), 'g', alpha = 0.1)
    

plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[i+1, a : a+51]), 'r', alpha = 0.5, label = 'downhill')
plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[i+1, a : a+51]), 'b', alpha = 0.5, label = 'flat')
plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[i+1, a : a+51]), 'g', alpha = 0.5, label = 'uphill')
    
plt.legend()
plt.title("Right Thigh Extension Angle - Mean ("+ str(df.columns[a]) + ")")
plt.xlabel("Percentage of Gait Cycle [%]")
plt.ylabel("Angle [deg]")
```





    <matplotlib.text.Text at 0x1429e6ac8>




![png](Report-IntroductionandEDA_files/Report-IntroductionandEDA_24_1.png)


#### Comments:
* When the thigh angle is centered around 0 degree, the main differences in the three classes can be shown at either near 0% or 100% gait cycle as well as around 40%
* The uphill signals seem most different than flat and downhill 



```python
a = 52+107 + 107 + 107

for i in range(100):
    plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[8*i, a : a+51]), 'r', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[8*i, a : a+51]), 'b', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[8*i, a : a+51]), 'g', alpha = 0.1)
    

plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[i+1, a : a+51]), 'r', alpha = 0.5, label = 'downhill')
plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[i+1, a : a+51]), 'b', alpha = 0.5, label = 'flat')
plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[i+1, a : a+51]), 'g', alpha = 0.5, label = 'uphill')
    
plt.legend()
plt.title("Right Thigh Angular Velocity (" + str(df.columns[a][:-2] + ")"))
plt.xlabel("Percentage of Gait Cycle [%]")
plt.ylabel("Angular Velocity [deg/s]")
```





    <matplotlib.text.Text at 0x1421de2b0>




![png](Report-IntroductionandEDA_files/Report-IntroductionandEDA_26_1.png)


#### Comments:
* Looking at the angular velocity of thigh extension, we see that the uphill features seem the most different from two other classes
* flat ground and downhill signals overlaps greatly, may be difficult to use this feature to classify



```python
a = 52+54+107+ 107+ 107

for i in range(100):
    plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[8*i, a : a+51]), 'r', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[8*i, a : a+51]), 'b', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[8*i, a : a+51]), 'g', alpha = 0.1)
    

plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[i+1, a : a+51]), 'r', alpha = 0.5, label = 'downhill')
plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[i+1, a : a+51]), 'b', alpha = 0.5, label = 'flat')
plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[i+1, a : a+51]), 'g', alpha = 0.5, label = 'uphill')
    
plt.legend()
plt.title("Thigh Angular Velocity  - Mean ("+ str(df.columns[a]) + ")")
plt.xlabel("Percentage of Gait Cycle [%]")
plt.ylabel("Angular Velocity [deg/s]")
```





    <matplotlib.text.Text at 0x1442da7b8>




![png](Report-IntroductionandEDA_files/Report-IntroductionandEDA_28_1.png)


#### Comments:
* Now that the angular velocity signal is centered around 0 deg/s, the variance and distribution of different classes do not seem so different
* Angular velocity may not be sensitive to different sensor placements (donning sensor descrepancies) 



```python
a = np.where(df_flat.columns == 'gyroZ_A_0')[0][0]

for i in range(100):
    plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[8*i, a : a+51]), 'r', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[8*i, a : a+51]), 'b', alpha = 0.1)
    plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[8*i, a : a+51]), 'g', alpha = 0.1)
    

plt.plot(np.arange(0, 102, 2), np.array(df_downhill.iloc[i+1, a : a+51]), 'r', alpha = 0.5, label = 'downhill')
plt.plot(np.arange(0, 102, 2), np.array(df_flat.iloc[i+1, a : a+51]), 'b', alpha = 0.5, label = 'flat')
plt.plot(np.arange(0, 102, 2), np.array(df_uphill.iloc[i+1, a : a+51]), 'g', alpha = 0.5, label = 'uphill')
    
plt.legend()
plt.title("Yaw Angular Velocity ("+ str(df.columns[a][:-2]) + ")")
plt.xlabel("Percentage of Gait Cycle [%]")
plt.ylabel("Angular Velocity [deg/s]")
```





    <matplotlib.text.Text at 0x152f62828>




![png](Report-IntroductionandEDA_files/Report-IntroductionandEDA_30_1.png)


#### Comments:
* The gyro in the z direction caputres some interesting signals during a gait cycle
* Uphill data has more extreme values
* Flat ground and downhill data have the most differences at around 10% and 55% gait cycle

### EDA for 7-Class Classification/Regression

Secondly, we performed some data exploration for the regression problem. Even though the data is collected   at   discrete   slopes,   the   relationship   between   certain   predictors   and   walking   slope   can still   be   shown.



```python
set(df['ref_slope'])
```





    {-10, -5, 0, 5, 10, 15, 20}





```python

df_n10 = df[df['ref_slope'] == -10]
df_n5 = df[df['ref_slope'] == -5]
df_0 = df[df['ref_slope'] == 0]
df_5 = df[df['ref_slope'] == 5]
df_10 = df[df['ref_slope'] == 10]
df_15 = df[df['ref_slope'] == 15]
df_20 = df[df['ref_slope'] == 20]


msk1 = np.random.rand(len(df_n10)) < 1000 /len(df_n10)
msk2 = np.random.rand(len(df_n5)) < 1000 /len(df_n5)
msk3 = np.random.rand(len(df_0)) < 1000 /len(df_0)
msk4 = np.random.rand(len(df_5)) < 1000 /len(df_5)
msk5 = np.random.rand(len(df_10)) < 1000 /len(df_10)
msk6 = np.random.rand(len(df_15)) < 1000 /len(df_15)
msk7 = np.random.rand(len(df_20)) < 1000 /len(df_20)

df_n10 = df_n10[msk1]
df_n5 = df_n5[msk2]
df_0 = df_0[msk3]
df_5 = df_5[msk4]
df_10 = df_10[msk5]
df_15 = df_15[msk6]
df_20 = df_20[msk7]

```




```python
! pip install colour

from colour import Color
red = Color("red")
colors = list(red.range_to(Color("green"),7))
```


    Requirement already satisfied: colour in /Users/kate_zym/anaconda/lib/python3.6/site-packages




```python

a = 52

from colour import Color
red = Color("red")
colors = list(red.range_to(Color("green"),10))

for i in range(100):
    plt.plot(np.arange(0, 102, 2), np.array(df_n10.iloc[4*i, a : a+51]), c = colors[0].get_rgb(), alpha = 0.4)
    plt.plot(np.arange(0, 102, 2), np.array(df_n5.iloc[4*i, a : a+51]), c = colors[1].get_rgb(), alpha = 0.4)
    plt.plot(np.arange(0, 102, 2), np.array(df_0.iloc[8*i, a : a+51]), c = colors[2].get_rgb(), alpha = 0.4)
    plt.plot(np.arange(0, 102, 2), np.array(df_5.iloc[8*i, a : a+51]), c = colors[3].get_rgb(), alpha = 0.4)
    plt.plot(np.arange(0, 102, 2), np.array(df_10.iloc[4*i, a : a+51]), c = colors[4].get_rgb(), alpha = 0.4)
    plt.plot(np.arange(0, 102, 2), np.array(df_15.iloc[4*i, a : a+51]), c = colors[5].get_rgb(), alpha = 0.4)
    plt.plot(np.arange(0, 102, 2), np.array(df_20.iloc[2*i, a : a+51]), c = colors[6].get_rgb(), alpha = 0.4)
    

plt.plot(np.arange(0, 102, 2), np.array(df_n10.iloc[4*i, a : a+51]), c = colors[0].get_rgb(), alpha = 1, label = 
    '-10')
plt.plot(np.arange(0, 102, 2), np.array(df_n5.iloc[4*i, a : a+51]), c = colors[1].get_rgb(), alpha = 1, label = '-5% Slope')
plt.plot(np.arange(0, 102, 2), np.array(df_0.iloc[8*i, a : a+51]), c = colors[2].get_rgb(), alpha = 1, label = '0% Slope')
plt.plot(np.arange(0, 102, 2), np.array(df_5.iloc[8*i, a : a+51]), c = colors[3].get_rgb(), alpha = 1, label = '5% Slope')
plt.plot(np.arange(0, 102, 2), np.array(df_10.iloc[4*i, a : a+51]), c = colors[4].get_rgb(), alpha = 1, label = '10% Slope')
plt.plot(np.arange(0, 102, 2), np.array(df_15.iloc[4*i, a : a+51]), c = colors[5].get_rgb(), alpha = 1, label = '15% Slope')
plt.plot(np.arange(0, 102, 2), np.array(df_20.iloc[2*i, a : a+51]), c = colors[6].get_rgb(), alpha = 1, label = '20% Slope')
    
plt.legend()
plt.title("Right Thigh Extension Angle (" + str(df.columns[a][:-2] + ")"))
plt.xlabel("Percentage of Gait Cycle [%]")
plt.ylabel("Angle [deg]")
```





    <matplotlib.text.Text at 0x123904780>




![png](Report-IntroductionandEDA_files/Report-IntroductionandEDA_37_1.png)


#### Comments:
* The thigh extension (roll) angles do seem to vary according to the different slope classes
* The most noticeable difference seem to be near the peak, around 38% GC



```python
sns.set_context("poster")
plt.plot(df['angleX_R_38'].values + df['angleX_L_38'].values, df['ref_slope'], 'bs', alpha= 0.3, label = 'Sum of Legs')

plt.xlabel('Roll Angle at 38% GC')
plt.ylabel('Reference Slope [%]')
plt.legend()
```





    <matplotlib.legend.Legend at 0x122f0d160>




![png](Report-IntroductionandEDA_files/Report-IntroductionandEDA_39_1.png)


#### Comments:
* Plotting the sum of thigh angles at 38% percent GC, we see that there is a trend of roll angle distribution at the different reference slopes
* with higher thigh angle values at 38% GC, there is a higher probability of walking uphill



```python
plt.scatter(df['angleX_R_max_mm'].values, df['ref_slope'], alpha= 0.1)

plt.xlabel('Maximum Roll Angle  minus Mean [deg]')
plt.ylabel('Reference Slope [%]')
plt.title('Slope vs. Roll Angle Amplitude')
```





    <matplotlib.text.Text at 0x1533ed470>




![png](Report-IntroductionandEDA_files/Report-IntroductionandEDA_41_1.png)


#### Comments:
* Similarly, we can look at the distribution of thigh angle amplitude
* The distribution and mean of thigh angle amplitude changes at different reference slopes
* As the thigh roll angle increases, the most probable reference slope also increases



```python

```




```python

```




```python

```




```python

```




```python
for col in df_flat.columns:
    print(col)
```


    subject_number
    time_0
    time_2
    time_4
    time_6
    time_8
    time_10
    time_12
    time_14
    time_16
    time_18
    time_20
    time_22
    time_24
    time_26
    time_28
    time_30
    time_32
    time_34
    time_36
    time_38
    time_40
    time_42
    time_44
    time_46
    time_48
    time_50
    time_52
    time_54
    time_56
    time_58
    time_60
    time_62
    time_64
    time_66
    time_68
    time_70
    time_72
    time_74
    time_76
    time_78
    time_80
    time_82
    time_84
    time_86
    time_88
    time_90
    time_92
    time_94
    time_96
    time_98
    time_100
    angleX_R_0
    angleX_R_2
    angleX_R_4
    angleX_R_6
    angleX_R_8
    angleX_R_10
    angleX_R_12
    angleX_R_14
    angleX_R_16
    angleX_R_18
    angleX_R_20
    angleX_R_22
    angleX_R_24
    angleX_R_26
    angleX_R_28
    angleX_R_30
    angleX_R_32
    angleX_R_34
    angleX_R_36
    angleX_R_38
    angleX_R_40
    angleX_R_42
    angleX_R_44
    angleX_R_46
    angleX_R_48
    angleX_R_50
    angleX_R_52
    angleX_R_54
    angleX_R_56
    angleX_R_58
    angleX_R_60
    angleX_R_62
    angleX_R_64
    angleX_R_66
    angleX_R_68
    angleX_R_70
    angleX_R_72
    angleX_R_74
    angleX_R_76
    angleX_R_78
    angleX_R_80
    angleX_R_82
    angleX_R_84
    angleX_R_86
    angleX_R_88
    angleX_R_90
    angleX_R_92
    angleX_R_94
    angleX_R_96
    angleX_R_98
    angleX_R_100
    angleX_R_max
    angleX_R_min
    angleX_R_mean
    angleX_R_0mm
    angleX_R_2mm
    angleX_R_4mm
    angleX_R_6mm
    angleX_R_8mm
    angleX_R_10mm
    angleX_R_12mm
    angleX_R_14mm
    angleX_R_16mm
    angleX_R_18mm
    angleX_R_20mm
    angleX_R_22mm
    angleX_R_24mm
    angleX_R_26mm
    angleX_R_28mm
    angleX_R_30mm
    angleX_R_32mm
    angleX_R_34mm
    angleX_R_36mm
    angleX_R_38mm
    angleX_R_40mm
    angleX_R_42mm
    angleX_R_44mm
    angleX_R_46mm
    angleX_R_48mm
    angleX_R_50mm
    angleX_R_52mm
    angleX_R_54mm
    angleX_R_56mm
    angleX_R_58mm
    angleX_R_60mm
    angleX_R_62mm
    angleX_R_64mm
    angleX_R_66mm
    angleX_R_68mm
    angleX_R_70mm
    angleX_R_72mm
    angleX_R_74mm
    angleX_R_76mm
    angleX_R_78mm
    angleX_R_80mm
    angleX_R_82mm
    angleX_R_84mm
    angleX_R_86mm
    angleX_R_88mm
    angleX_R_90mm
    angleX_R_92mm
    angleX_R_94mm
    angleX_R_96mm
    angleX_R_98mm
    angleX_R_100mm
    angleX_R_max_mm
    angleX_R_min_mm
    angleY_R_0
    angleY_R_2
    angleY_R_4
    angleY_R_6
    angleY_R_8
    angleY_R_10
    angleY_R_12
    angleY_R_14
    angleY_R_16
    angleY_R_18
    angleY_R_20
    angleY_R_22
    angleY_R_24
    angleY_R_26
    angleY_R_28
    angleY_R_30
    angleY_R_32
    angleY_R_34
    angleY_R_36
    angleY_R_38
    angleY_R_40
    angleY_R_42
    angleY_R_44
    angleY_R_46
    angleY_R_48
    angleY_R_50
    angleY_R_52
    angleY_R_54
    angleY_R_56
    angleY_R_58
    angleY_R_60
    angleY_R_62
    angleY_R_64
    angleY_R_66
    angleY_R_68
    angleY_R_70
    angleY_R_72
    angleY_R_74
    angleY_R_76
    angleY_R_78
    angleY_R_80
    angleY_R_82
    angleY_R_84
    angleY_R_86
    angleY_R_88
    angleY_R_90
    angleY_R_92
    angleY_R_94
    angleY_R_96
    angleY_R_98
    angleY_R_100
    angleY_R_max
    angleY_R_min
    angleY_R_mean
    angleY_R_0mm
    angleY_R_2mm
    angleY_R_4mm
    angleY_R_6mm
    angleY_R_8mm
    angleY_R_10mm
    angleY_R_12mm
    angleY_R_14mm
    angleY_R_16mm
    angleY_R_18mm
    angleY_R_20mm
    angleY_R_22mm
    angleY_R_24mm
    angleY_R_26mm
    angleY_R_28mm
    angleY_R_30mm
    angleY_R_32mm
    angleY_R_34mm
    angleY_R_36mm
    angleY_R_38mm
    angleY_R_40mm
    angleY_R_42mm
    angleY_R_44mm
    angleY_R_46mm
    angleY_R_48mm
    angleY_R_50mm
    angleY_R_52mm
    angleY_R_54mm
    angleY_R_56mm
    angleY_R_58mm
    angleY_R_60mm
    angleY_R_62mm
    angleY_R_64mm
    angleY_R_66mm
    angleY_R_68mm
    angleY_R_70mm
    angleY_R_72mm
    angleY_R_74mm
    angleY_R_76mm
    angleY_R_78mm
    angleY_R_80mm
    angleY_R_82mm
    angleY_R_84mm
    angleY_R_86mm
    angleY_R_88mm
    angleY_R_90mm
    angleY_R_92mm
    angleY_R_94mm
    angleY_R_96mm
    angleY_R_98mm
    angleY_R_100mm
    angleY_R_max_mm
    angleY_R_min_mm
    angleZ_R_0
    angleZ_R_2
    angleZ_R_4
    angleZ_R_6
    angleZ_R_8
    angleZ_R_10
    angleZ_R_12
    angleZ_R_14
    angleZ_R_16
    angleZ_R_18
    angleZ_R_20
    angleZ_R_22
    angleZ_R_24
    angleZ_R_26
    angleZ_R_28
    angleZ_R_30
    angleZ_R_32
    angleZ_R_34
    angleZ_R_36
    angleZ_R_38
    angleZ_R_40
    angleZ_R_42
    angleZ_R_44
    angleZ_R_46
    angleZ_R_48
    angleZ_R_50
    angleZ_R_52
    angleZ_R_54
    angleZ_R_56
    angleZ_R_58
    angleZ_R_60
    angleZ_R_62
    angleZ_R_64
    angleZ_R_66
    angleZ_R_68
    angleZ_R_70
    angleZ_R_72
    angleZ_R_74
    angleZ_R_76
    angleZ_R_78
    angleZ_R_80
    angleZ_R_82
    angleZ_R_84
    angleZ_R_86
    angleZ_R_88
    angleZ_R_90
    angleZ_R_92
    angleZ_R_94
    angleZ_R_96
    angleZ_R_98
    angleZ_R_100
    angleZ_R_max
    angleZ_R_min
    angleZ_R_mean
    angleZ_R_0mm
    angleZ_R_2mm
    angleZ_R_4mm
    angleZ_R_6mm
    angleZ_R_8mm
    angleZ_R_10mm
    angleZ_R_12mm
    angleZ_R_14mm
    angleZ_R_16mm
    angleZ_R_18mm
    angleZ_R_20mm
    angleZ_R_22mm
    angleZ_R_24mm
    angleZ_R_26mm
    angleZ_R_28mm
    angleZ_R_30mm
    angleZ_R_32mm
    angleZ_R_34mm
    angleZ_R_36mm
    angleZ_R_38mm
    angleZ_R_40mm
    angleZ_R_42mm
    angleZ_R_44mm
    angleZ_R_46mm
    angleZ_R_48mm
    angleZ_R_50mm
    angleZ_R_52mm
    angleZ_R_54mm
    angleZ_R_56mm
    angleZ_R_58mm
    angleZ_R_60mm
    angleZ_R_62mm
    angleZ_R_64mm
    angleZ_R_66mm
    angleZ_R_68mm
    angleZ_R_70mm
    angleZ_R_72mm
    angleZ_R_74mm
    angleZ_R_76mm
    angleZ_R_78mm
    angleZ_R_80mm
    angleZ_R_82mm
    angleZ_R_84mm
    angleZ_R_86mm
    angleZ_R_88mm
    angleZ_R_90mm
    angleZ_R_92mm
    angleZ_R_94mm
    angleZ_R_96mm
    angleZ_R_98mm
    angleZ_R_100mm
    angleZ_R_max_mm
    angleZ_R_min_mm
    gyroX_R_0
    gyroX_R_2
    gyroX_R_4
    gyroX_R_6
    gyroX_R_8
    gyroX_R_10
    gyroX_R_12
    gyroX_R_14
    gyroX_R_16
    gyroX_R_18
    gyroX_R_20
    gyroX_R_22
    gyroX_R_24
    gyroX_R_26
    gyroX_R_28
    gyroX_R_30
    gyroX_R_32
    gyroX_R_34
    gyroX_R_36
    gyroX_R_38
    gyroX_R_40
    gyroX_R_42
    gyroX_R_44
    gyroX_R_46
    gyroX_R_48
    gyroX_R_50
    gyroX_R_52
    gyroX_R_54
    gyroX_R_56
    gyroX_R_58
    gyroX_R_60
    gyroX_R_62
    gyroX_R_64
    gyroX_R_66
    gyroX_R_68
    gyroX_R_70
    gyroX_R_72
    gyroX_R_74
    gyroX_R_76
    gyroX_R_78
    gyroX_R_80
    gyroX_R_82
    gyroX_R_84
    gyroX_R_86
    gyroX_R_88
    gyroX_R_90
    gyroX_R_92
    gyroX_R_94
    gyroX_R_96
    gyroX_R_98
    gyroX_R_100
    gyroX_R_max
    gyroX_R_min
    gyroX_R_mean
    gyroX_R_0mm
    gyroX_R_2mm
    gyroX_R_4mm
    gyroX_R_6mm
    gyroX_R_8mm
    gyroX_R_10mm
    gyroX_R_12mm
    gyroX_R_14mm
    gyroX_R_16mm
    gyroX_R_18mm
    gyroX_R_20mm
    gyroX_R_22mm
    gyroX_R_24mm
    gyroX_R_26mm
    gyroX_R_28mm
    gyroX_R_30mm
    gyroX_R_32mm
    gyroX_R_34mm
    gyroX_R_36mm
    gyroX_R_38mm
    gyroX_R_40mm
    gyroX_R_42mm
    gyroX_R_44mm
    gyroX_R_46mm
    gyroX_R_48mm
    gyroX_R_50mm
    gyroX_R_52mm
    gyroX_R_54mm
    gyroX_R_56mm
    gyroX_R_58mm
    gyroX_R_60mm
    gyroX_R_62mm
    gyroX_R_64mm
    gyroX_R_66mm
    gyroX_R_68mm
    gyroX_R_70mm
    gyroX_R_72mm
    gyroX_R_74mm
    gyroX_R_76mm
    gyroX_R_78mm
    gyroX_R_80mm
    gyroX_R_82mm
    gyroX_R_84mm
    gyroX_R_86mm
    gyroX_R_88mm
    gyroX_R_90mm
    gyroX_R_92mm
    gyroX_R_94mm
    gyroX_R_96mm
    gyroX_R_98mm
    gyroX_R_100mm
    gyroX_R_max_mm
    gyroX_R_min_mm
    gyroY_R_0
    gyroY_R_2
    gyroY_R_4
    gyroY_R_6
    gyroY_R_8
    gyroY_R_10
    gyroY_R_12
    gyroY_R_14
    gyroY_R_16
    gyroY_R_18
    gyroY_R_20
    gyroY_R_22
    gyroY_R_24
    gyroY_R_26
    gyroY_R_28
    gyroY_R_30
    gyroY_R_32
    gyroY_R_34
    gyroY_R_36
    gyroY_R_38
    gyroY_R_40
    gyroY_R_42
    gyroY_R_44
    gyroY_R_46
    gyroY_R_48
    gyroY_R_50
    gyroY_R_52
    gyroY_R_54
    gyroY_R_56
    gyroY_R_58
    gyroY_R_60
    gyroY_R_62
    gyroY_R_64
    gyroY_R_66
    gyroY_R_68
    gyroY_R_70
    gyroY_R_72
    gyroY_R_74
    gyroY_R_76
    gyroY_R_78
    gyroY_R_80
    gyroY_R_82
    gyroY_R_84
    gyroY_R_86
    gyroY_R_88
    gyroY_R_90
    gyroY_R_92
    gyroY_R_94
    gyroY_R_96
    gyroY_R_98
    gyroY_R_100
    gyroY_R_max
    gyroY_R_min
    gyroY_R_mean
    gyroY_R_0mm
    gyroY_R_2mm
    gyroY_R_4mm
    gyroY_R_6mm
    gyroY_R_8mm
    gyroY_R_10mm
    gyroY_R_12mm
    gyroY_R_14mm
    gyroY_R_16mm
    gyroY_R_18mm
    gyroY_R_20mm
    gyroY_R_22mm
    gyroY_R_24mm
    gyroY_R_26mm
    gyroY_R_28mm
    gyroY_R_30mm
    gyroY_R_32mm
    gyroY_R_34mm
    gyroY_R_36mm
    gyroY_R_38mm
    gyroY_R_40mm
    gyroY_R_42mm
    gyroY_R_44mm
    gyroY_R_46mm
    gyroY_R_48mm
    gyroY_R_50mm
    gyroY_R_52mm
    gyroY_R_54mm
    gyroY_R_56mm
    gyroY_R_58mm
    gyroY_R_60mm
    gyroY_R_62mm
    gyroY_R_64mm
    gyroY_R_66mm
    gyroY_R_68mm
    gyroY_R_70mm
    gyroY_R_72mm
    gyroY_R_74mm
    gyroY_R_76mm
    gyroY_R_78mm
    gyroY_R_80mm
    gyroY_R_82mm
    gyroY_R_84mm
    gyroY_R_86mm
    gyroY_R_88mm
    gyroY_R_90mm
    gyroY_R_92mm
    gyroY_R_94mm
    gyroY_R_96mm
    gyroY_R_98mm
    gyroY_R_100mm
    gyroY_R_max_mm
    gyroY_R_min_mm
    gyroZ_R_0
    gyroZ_R_2
    gyroZ_R_4
    gyroZ_R_6
    gyroZ_R_8
    gyroZ_R_10
    gyroZ_R_12
    gyroZ_R_14
    gyroZ_R_16
    gyroZ_R_18
    gyroZ_R_20
    gyroZ_R_22
    gyroZ_R_24
    gyroZ_R_26
    gyroZ_R_28
    gyroZ_R_30
    gyroZ_R_32
    gyroZ_R_34
    gyroZ_R_36
    gyroZ_R_38
    gyroZ_R_40
    gyroZ_R_42
    gyroZ_R_44
    gyroZ_R_46
    gyroZ_R_48
    gyroZ_R_50
    gyroZ_R_52
    gyroZ_R_54
    gyroZ_R_56
    gyroZ_R_58
    gyroZ_R_60
    gyroZ_R_62
    gyroZ_R_64
    gyroZ_R_66
    gyroZ_R_68
    gyroZ_R_70
    gyroZ_R_72
    gyroZ_R_74
    gyroZ_R_76
    gyroZ_R_78
    gyroZ_R_80
    gyroZ_R_82
    gyroZ_R_84
    gyroZ_R_86
    gyroZ_R_88
    gyroZ_R_90
    gyroZ_R_92
    gyroZ_R_94
    gyroZ_R_96
    gyroZ_R_98
    gyroZ_R_100
    gyroZ_R_max
    gyroZ_R_min
    gyroZ_R_mean
    gyroZ_R_0mm
    gyroZ_R_2mm
    gyroZ_R_4mm
    gyroZ_R_6mm
    gyroZ_R_8mm
    gyroZ_R_10mm
    gyroZ_R_12mm
    gyroZ_R_14mm
    gyroZ_R_16mm
    gyroZ_R_18mm
    gyroZ_R_20mm
    gyroZ_R_22mm
    gyroZ_R_24mm
    gyroZ_R_26mm
    gyroZ_R_28mm
    gyroZ_R_30mm
    gyroZ_R_32mm
    gyroZ_R_34mm
    gyroZ_R_36mm
    gyroZ_R_38mm
    gyroZ_R_40mm
    gyroZ_R_42mm
    gyroZ_R_44mm
    gyroZ_R_46mm
    gyroZ_R_48mm
    gyroZ_R_50mm
    gyroZ_R_52mm
    gyroZ_R_54mm
    gyroZ_R_56mm
    gyroZ_R_58mm
    gyroZ_R_60mm
    gyroZ_R_62mm
    gyroZ_R_64mm
    gyroZ_R_66mm
    gyroZ_R_68mm
    gyroZ_R_70mm
    gyroZ_R_72mm
    gyroZ_R_74mm
    gyroZ_R_76mm
    gyroZ_R_78mm
    gyroZ_R_80mm
    gyroZ_R_82mm
    gyroZ_R_84mm
    gyroZ_R_86mm
    gyroZ_R_88mm
    gyroZ_R_90mm
    gyroZ_R_92mm
    gyroZ_R_94mm
    gyroZ_R_96mm
    gyroZ_R_98mm
    gyroZ_R_100mm
    gyroZ_R_max_mm
    gyroZ_R_min_mm
    accXRot_R_0
    accXRot_R_2
    accXRot_R_4
    accXRot_R_6
    accXRot_R_8
    accXRot_R_10
    accXRot_R_12
    accXRot_R_14
    accXRot_R_16
    accXRot_R_18
    accXRot_R_20
    accXRot_R_22
    accXRot_R_24
    accXRot_R_26
    accXRot_R_28
    accXRot_R_30
    accXRot_R_32
    accXRot_R_34
    accXRot_R_36
    accXRot_R_38
    accXRot_R_40
    accXRot_R_42
    accXRot_R_44
    accXRot_R_46
    accXRot_R_48
    accXRot_R_50
    accXRot_R_52
    accXRot_R_54
    accXRot_R_56
    accXRot_R_58
    accXRot_R_60
    accXRot_R_62
    accXRot_R_64
    accXRot_R_66
    accXRot_R_68
    accXRot_R_70
    accXRot_R_72
    accXRot_R_74
    accXRot_R_76
    accXRot_R_78
    accXRot_R_80
    accXRot_R_82
    accXRot_R_84
    accXRot_R_86
    accXRot_R_88
    accXRot_R_90
    accXRot_R_92
    accXRot_R_94
    accXRot_R_96
    accXRot_R_98
    accXRot_R_100
    accXRot_R_max
    accXRot_R_min
    accXRot_R_mean
    accXRot_R_0mm
    accXRot_R_2mm
    accXRot_R_4mm
    accXRot_R_6mm
    accXRot_R_8mm
    accXRot_R_10mm
    accXRot_R_12mm
    accXRot_R_14mm
    accXRot_R_16mm
    accXRot_R_18mm
    accXRot_R_20mm
    accXRot_R_22mm
    accXRot_R_24mm
    accXRot_R_26mm
    accXRot_R_28mm
    accXRot_R_30mm
    accXRot_R_32mm
    accXRot_R_34mm
    accXRot_R_36mm
    accXRot_R_38mm
    accXRot_R_40mm
    accXRot_R_42mm
    accXRot_R_44mm
    accXRot_R_46mm
    accXRot_R_48mm
    accXRot_R_50mm
    accXRot_R_52mm
    accXRot_R_54mm
    accXRot_R_56mm
    accXRot_R_58mm
    accXRot_R_60mm
    accXRot_R_62mm
    accXRot_R_64mm
    accXRot_R_66mm
    accXRot_R_68mm
    accXRot_R_70mm
    accXRot_R_72mm
    accXRot_R_74mm
    accXRot_R_76mm
    accXRot_R_78mm
    accXRot_R_80mm
    accXRot_R_82mm
    accXRot_R_84mm
    accXRot_R_86mm
    accXRot_R_88mm
    accXRot_R_90mm
    accXRot_R_92mm
    accXRot_R_94mm
    accXRot_R_96mm
    accXRot_R_98mm
    accXRot_R_100mm
    accXRot_R_max_mm
    accXRot_R_min_mm
    accYRot_R_0
    accYRot_R_2
    accYRot_R_4
    accYRot_R_6
    accYRot_R_8
    accYRot_R_10
    accYRot_R_12
    accYRot_R_14
    accYRot_R_16
    accYRot_R_18
    accYRot_R_20
    accYRot_R_22
    accYRot_R_24
    accYRot_R_26
    accYRot_R_28
    accYRot_R_30
    accYRot_R_32
    accYRot_R_34
    accYRot_R_36
    accYRot_R_38
    accYRot_R_40
    accYRot_R_42
    accYRot_R_44
    accYRot_R_46
    accYRot_R_48
    accYRot_R_50
    accYRot_R_52
    accYRot_R_54
    accYRot_R_56
    accYRot_R_58
    accYRot_R_60
    accYRot_R_62
    accYRot_R_64
    accYRot_R_66
    accYRot_R_68
    accYRot_R_70
    accYRot_R_72
    accYRot_R_74
    accYRot_R_76
    accYRot_R_78
    accYRot_R_80
    accYRot_R_82
    accYRot_R_84
    accYRot_R_86
    accYRot_R_88
    accYRot_R_90
    accYRot_R_92
    accYRot_R_94
    accYRot_R_96
    accYRot_R_98
    accYRot_R_100
    accYRot_R_max
    accYRot_R_min
    accYRot_R_mean
    accYRot_R_0mm
    accYRot_R_2mm
    accYRot_R_4mm
    accYRot_R_6mm
    accYRot_R_8mm
    accYRot_R_10mm
    accYRot_R_12mm
    accYRot_R_14mm
    accYRot_R_16mm
    accYRot_R_18mm
    accYRot_R_20mm
    accYRot_R_22mm
    accYRot_R_24mm
    accYRot_R_26mm
    accYRot_R_28mm
    accYRot_R_30mm
    accYRot_R_32mm
    accYRot_R_34mm
    accYRot_R_36mm
    accYRot_R_38mm
    accYRot_R_40mm
    accYRot_R_42mm
    accYRot_R_44mm
    accYRot_R_46mm
    accYRot_R_48mm
    accYRot_R_50mm
    accYRot_R_52mm
    accYRot_R_54mm
    accYRot_R_56mm
    accYRot_R_58mm
    accYRot_R_60mm
    accYRot_R_62mm
    accYRot_R_64mm
    accYRot_R_66mm
    accYRot_R_68mm
    accYRot_R_70mm
    accYRot_R_72mm
    accYRot_R_74mm
    accYRot_R_76mm
    accYRot_R_78mm
    accYRot_R_80mm
    accYRot_R_82mm
    accYRot_R_84mm
    accYRot_R_86mm
    accYRot_R_88mm
    accYRot_R_90mm
    accYRot_R_92mm
    accYRot_R_94mm
    accYRot_R_96mm
    accYRot_R_98mm
    accYRot_R_100mm
    accYRot_R_max_mm
    accYRot_R_min_mm
    accZRot_R_0
    accZRot_R_2
    accZRot_R_4
    accZRot_R_6
    accZRot_R_8
    accZRot_R_10
    accZRot_R_12
    accZRot_R_14
    accZRot_R_16
    accZRot_R_18
    accZRot_R_20
    accZRot_R_22
    accZRot_R_24
    accZRot_R_26
    accZRot_R_28
    accZRot_R_30
    accZRot_R_32
    accZRot_R_34
    accZRot_R_36
    accZRot_R_38
    accZRot_R_40
    accZRot_R_42
    accZRot_R_44
    accZRot_R_46
    accZRot_R_48
    accZRot_R_50
    accZRot_R_52
    accZRot_R_54
    accZRot_R_56
    accZRot_R_58
    accZRot_R_60
    accZRot_R_62
    accZRot_R_64
    accZRot_R_66
    accZRot_R_68
    accZRot_R_70
    accZRot_R_72
    accZRot_R_74
    accZRot_R_76
    accZRot_R_78
    accZRot_R_80
    accZRot_R_82
    accZRot_R_84
    accZRot_R_86
    accZRot_R_88
    accZRot_R_90
    accZRot_R_92
    accZRot_R_94
    accZRot_R_96
    accZRot_R_98
    accZRot_R_100
    accZRot_R_max
    accZRot_R_min
    accZRot_R_mean
    accZRot_R_0mm
    accZRot_R_2mm
    accZRot_R_4mm
    accZRot_R_6mm
    accZRot_R_8mm
    accZRot_R_10mm
    accZRot_R_12mm
    accZRot_R_14mm
    accZRot_R_16mm
    accZRot_R_18mm
    accZRot_R_20mm
    accZRot_R_22mm
    accZRot_R_24mm
    accZRot_R_26mm
    accZRot_R_28mm
    accZRot_R_30mm
    accZRot_R_32mm
    accZRot_R_34mm
    accZRot_R_36mm
    accZRot_R_38mm
    accZRot_R_40mm
    accZRot_R_42mm
    accZRot_R_44mm
    accZRot_R_46mm
    accZRot_R_48mm
    accZRot_R_50mm
    accZRot_R_52mm
    accZRot_R_54mm
    accZRot_R_56mm
    accZRot_R_58mm
    accZRot_R_60mm
    accZRot_R_62mm
    accZRot_R_64mm
    accZRot_R_66mm
    accZRot_R_68mm
    accZRot_R_70mm
    accZRot_R_72mm
    accZRot_R_74mm
    accZRot_R_76mm
    accZRot_R_78mm
    accZRot_R_80mm
    accZRot_R_82mm
    accZRot_R_84mm
    accZRot_R_86mm
    accZRot_R_88mm
    accZRot_R_90mm
    accZRot_R_92mm
    accZRot_R_94mm
    accZRot_R_96mm
    accZRot_R_98mm
    accZRot_R_100mm
    accZRot_R_max_mm
    accZRot_R_min_mm
    angleX_L_0
    angleX_L_2
    angleX_L_4
    angleX_L_6
    angleX_L_8
    angleX_L_10
    angleX_L_12
    angleX_L_14
    angleX_L_16
    angleX_L_18
    angleX_L_20
    angleX_L_22
    angleX_L_24
    angleX_L_26
    angleX_L_28
    angleX_L_30
    angleX_L_32
    angleX_L_34
    angleX_L_36
    angleX_L_38
    angleX_L_40
    angleX_L_42
    angleX_L_44
    angleX_L_46
    angleX_L_48
    angleX_L_50
    angleX_L_52
    angleX_L_54
    angleX_L_56
    angleX_L_58
    angleX_L_60
    angleX_L_62
    angleX_L_64
    angleX_L_66
    angleX_L_68
    angleX_L_70
    angleX_L_72
    angleX_L_74
    angleX_L_76
    angleX_L_78
    angleX_L_80
    angleX_L_82
    angleX_L_84
    angleX_L_86
    angleX_L_88
    angleX_L_90
    angleX_L_92
    angleX_L_94
    angleX_L_96
    angleX_L_98
    angleX_L_100
    angleX_L_max
    angleX_L_min
    angleX_L_mean
    angleX_L_0mm
    angleX_L_2mm
    angleX_L_4mm
    angleX_L_6mm
    angleX_L_8mm
    angleX_L_10mm
    angleX_L_12mm
    angleX_L_14mm
    angleX_L_16mm
    angleX_L_18mm
    angleX_L_20mm
    angleX_L_22mm
    angleX_L_24mm
    angleX_L_26mm
    angleX_L_28mm
    angleX_L_30mm
    angleX_L_32mm
    angleX_L_34mm
    angleX_L_36mm
    angleX_L_38mm
    angleX_L_40mm
    angleX_L_42mm
    angleX_L_44mm
    angleX_L_46mm
    angleX_L_48mm
    angleX_L_50mm
    angleX_L_52mm
    angleX_L_54mm
    angleX_L_56mm
    angleX_L_58mm
    angleX_L_60mm
    angleX_L_62mm
    angleX_L_64mm
    angleX_L_66mm
    angleX_L_68mm
    angleX_L_70mm
    angleX_L_72mm
    angleX_L_74mm
    angleX_L_76mm
    angleX_L_78mm
    angleX_L_80mm
    angleX_L_82mm
    angleX_L_84mm
    angleX_L_86mm
    angleX_L_88mm
    angleX_L_90mm
    angleX_L_92mm
    angleX_L_94mm
    angleX_L_96mm
    angleX_L_98mm
    angleX_L_100mm
    angleX_L_max_mm
    angleX_L_min_mm
    angleY_L_0
    angleY_L_2
    angleY_L_4
    angleY_L_6
    angleY_L_8
    angleY_L_10
    angleY_L_12
    angleY_L_14
    angleY_L_16
    angleY_L_18
    angleY_L_20
    angleY_L_22
    angleY_L_24
    angleY_L_26
    angleY_L_28
    angleY_L_30
    angleY_L_32
    angleY_L_34
    angleY_L_36
    angleY_L_38
    angleY_L_40
    angleY_L_42
    angleY_L_44
    angleY_L_46
    angleY_L_48
    angleY_L_50
    angleY_L_52
    angleY_L_54
    angleY_L_56
    angleY_L_58
    angleY_L_60
    angleY_L_62
    angleY_L_64
    angleY_L_66
    angleY_L_68
    angleY_L_70
    angleY_L_72
    angleY_L_74
    angleY_L_76
    angleY_L_78
    angleY_L_80
    angleY_L_82
    angleY_L_84
    angleY_L_86
    angleY_L_88
    angleY_L_90
    angleY_L_92
    angleY_L_94
    angleY_L_96
    angleY_L_98
    angleY_L_100
    angleY_L_max
    angleY_L_min
    angleY_L_mean
    angleY_L_0mm
    angleY_L_2mm
    angleY_L_4mm
    angleY_L_6mm
    angleY_L_8mm
    angleY_L_10mm
    angleY_L_12mm
    angleY_L_14mm
    angleY_L_16mm
    angleY_L_18mm
    angleY_L_20mm
    angleY_L_22mm
    angleY_L_24mm
    angleY_L_26mm
    angleY_L_28mm
    angleY_L_30mm
    angleY_L_32mm
    angleY_L_34mm
    angleY_L_36mm
    angleY_L_38mm
    angleY_L_40mm
    angleY_L_42mm
    angleY_L_44mm
    angleY_L_46mm
    angleY_L_48mm
    angleY_L_50mm
    angleY_L_52mm
    angleY_L_54mm
    angleY_L_56mm
    angleY_L_58mm
    angleY_L_60mm
    angleY_L_62mm
    angleY_L_64mm
    angleY_L_66mm
    angleY_L_68mm
    angleY_L_70mm
    angleY_L_72mm
    angleY_L_74mm
    angleY_L_76mm
    angleY_L_78mm
    angleY_L_80mm
    angleY_L_82mm
    angleY_L_84mm
    angleY_L_86mm
    angleY_L_88mm
    angleY_L_90mm
    angleY_L_92mm
    angleY_L_94mm
    angleY_L_96mm
    angleY_L_98mm
    angleY_L_100mm
    angleY_L_max_mm
    angleY_L_min_mm
    angleZ_L_0
    angleZ_L_2
    angleZ_L_4
    angleZ_L_6
    angleZ_L_8
    angleZ_L_10
    angleZ_L_12
    angleZ_L_14
    angleZ_L_16
    angleZ_L_18
    angleZ_L_20
    angleZ_L_22
    angleZ_L_24
    angleZ_L_26
    angleZ_L_28
    angleZ_L_30
    angleZ_L_32
    angleZ_L_34
    angleZ_L_36
    angleZ_L_38
    angleZ_L_40
    angleZ_L_42
    angleZ_L_44
    angleZ_L_46
    angleZ_L_48
    angleZ_L_50
    angleZ_L_52
    angleZ_L_54
    angleZ_L_56
    angleZ_L_58
    angleZ_L_60
    angleZ_L_62
    angleZ_L_64
    angleZ_L_66
    angleZ_L_68
    angleZ_L_70
    angleZ_L_72
    angleZ_L_74
    angleZ_L_76
    angleZ_L_78
    angleZ_L_80
    angleZ_L_82
    angleZ_L_84
    angleZ_L_86
    angleZ_L_88
    angleZ_L_90
    angleZ_L_92
    angleZ_L_94
    angleZ_L_96
    angleZ_L_98
    angleZ_L_100
    angleZ_L_max
    angleZ_L_min
    angleZ_L_mean
    angleZ_L_0mm
    angleZ_L_2mm
    angleZ_L_4mm
    angleZ_L_6mm
    angleZ_L_8mm
    angleZ_L_10mm
    angleZ_L_12mm
    angleZ_L_14mm
    angleZ_L_16mm
    angleZ_L_18mm
    angleZ_L_20mm
    angleZ_L_22mm
    angleZ_L_24mm
    angleZ_L_26mm
    angleZ_L_28mm
    angleZ_L_30mm
    angleZ_L_32mm
    angleZ_L_34mm
    angleZ_L_36mm
    angleZ_L_38mm
    angleZ_L_40mm
    angleZ_L_42mm
    angleZ_L_44mm
    angleZ_L_46mm
    angleZ_L_48mm
    angleZ_L_50mm
    angleZ_L_52mm
    angleZ_L_54mm
    angleZ_L_56mm
    angleZ_L_58mm
    angleZ_L_60mm
    angleZ_L_62mm
    angleZ_L_64mm
    angleZ_L_66mm
    angleZ_L_68mm
    angleZ_L_70mm
    angleZ_L_72mm
    angleZ_L_74mm
    angleZ_L_76mm
    angleZ_L_78mm
    angleZ_L_80mm
    angleZ_L_82mm
    angleZ_L_84mm
    angleZ_L_86mm
    angleZ_L_88mm
    angleZ_L_90mm
    angleZ_L_92mm
    angleZ_L_94mm
    angleZ_L_96mm
    angleZ_L_98mm
    angleZ_L_100mm
    angleZ_L_max_mm
    angleZ_L_min_mm
    gyroX_L_0
    gyroX_L_2
    gyroX_L_4
    gyroX_L_6
    gyroX_L_8
    gyroX_L_10
    gyroX_L_12
    gyroX_L_14
    gyroX_L_16
    gyroX_L_18
    gyroX_L_20
    gyroX_L_22
    gyroX_L_24
    gyroX_L_26
    gyroX_L_28
    gyroX_L_30
    gyroX_L_32
    gyroX_L_34
    gyroX_L_36
    gyroX_L_38
    gyroX_L_40
    gyroX_L_42
    gyroX_L_44
    gyroX_L_46
    gyroX_L_48
    gyroX_L_50
    gyroX_L_52
    gyroX_L_54
    gyroX_L_56
    gyroX_L_58
    gyroX_L_60
    gyroX_L_62
    gyroX_L_64
    gyroX_L_66
    gyroX_L_68
    gyroX_L_70
    gyroX_L_72
    gyroX_L_74
    gyroX_L_76
    gyroX_L_78
    gyroX_L_80
    gyroX_L_82
    gyroX_L_84
    gyroX_L_86
    gyroX_L_88
    gyroX_L_90
    gyroX_L_92
    gyroX_L_94
    gyroX_L_96
    gyroX_L_98
    gyroX_L_100
    gyroX_L_max
    gyroX_L_min
    gyroX_L_mean
    gyroX_L_0mm
    gyroX_L_2mm
    gyroX_L_4mm
    gyroX_L_6mm
    gyroX_L_8mm
    gyroX_L_10mm
    gyroX_L_12mm
    gyroX_L_14mm
    gyroX_L_16mm
    gyroX_L_18mm
    gyroX_L_20mm
    gyroX_L_22mm
    gyroX_L_24mm
    gyroX_L_26mm
    gyroX_L_28mm
    gyroX_L_30mm
    gyroX_L_32mm
    gyroX_L_34mm
    gyroX_L_36mm
    gyroX_L_38mm
    gyroX_L_40mm
    gyroX_L_42mm
    gyroX_L_44mm
    gyroX_L_46mm
    gyroX_L_48mm
    gyroX_L_50mm
    gyroX_L_52mm
    gyroX_L_54mm
    gyroX_L_56mm
    gyroX_L_58mm
    gyroX_L_60mm
    gyroX_L_62mm
    gyroX_L_64mm
    gyroX_L_66mm
    gyroX_L_68mm
    gyroX_L_70mm
    gyroX_L_72mm
    gyroX_L_74mm
    gyroX_L_76mm
    gyroX_L_78mm
    gyroX_L_80mm
    gyroX_L_82mm
    gyroX_L_84mm
    gyroX_L_86mm
    gyroX_L_88mm
    gyroX_L_90mm
    gyroX_L_92mm
    gyroX_L_94mm
    gyroX_L_96mm
    gyroX_L_98mm
    gyroX_L_100mm
    gyroX_L_max_mm
    gyroX_L_min_mm
    gyroY_L_0
    gyroY_L_2
    gyroY_L_4
    gyroY_L_6
    gyroY_L_8
    gyroY_L_10
    gyroY_L_12
    gyroY_L_14
    gyroY_L_16
    gyroY_L_18
    gyroY_L_20
    gyroY_L_22
    gyroY_L_24
    gyroY_L_26
    gyroY_L_28
    gyroY_L_30
    gyroY_L_32
    gyroY_L_34
    gyroY_L_36
    gyroY_L_38
    gyroY_L_40
    gyroY_L_42
    gyroY_L_44
    gyroY_L_46
    gyroY_L_48
    gyroY_L_50
    gyroY_L_52
    gyroY_L_54
    gyroY_L_56
    gyroY_L_58
    gyroY_L_60
    gyroY_L_62
    gyroY_L_64
    gyroY_L_66
    gyroY_L_68
    gyroY_L_70
    gyroY_L_72
    gyroY_L_74
    gyroY_L_76
    gyroY_L_78
    gyroY_L_80
    gyroY_L_82
    gyroY_L_84
    gyroY_L_86
    gyroY_L_88
    gyroY_L_90
    gyroY_L_92
    gyroY_L_94
    gyroY_L_96
    gyroY_L_98
    gyroY_L_100
    gyroY_L_max
    gyroY_L_min
    gyroY_L_mean
    gyroY_L_0mm
    gyroY_L_2mm
    gyroY_L_4mm
    gyroY_L_6mm
    gyroY_L_8mm
    gyroY_L_10mm
    gyroY_L_12mm
    gyroY_L_14mm
    gyroY_L_16mm
    gyroY_L_18mm
    gyroY_L_20mm
    gyroY_L_22mm
    gyroY_L_24mm
    gyroY_L_26mm
    gyroY_L_28mm
    gyroY_L_30mm
    gyroY_L_32mm
    gyroY_L_34mm
    gyroY_L_36mm
    gyroY_L_38mm
    gyroY_L_40mm
    gyroY_L_42mm
    gyroY_L_44mm
    gyroY_L_46mm
    gyroY_L_48mm
    gyroY_L_50mm
    gyroY_L_52mm
    gyroY_L_54mm
    gyroY_L_56mm
    gyroY_L_58mm
    gyroY_L_60mm
    gyroY_L_62mm
    gyroY_L_64mm
    gyroY_L_66mm
    gyroY_L_68mm
    gyroY_L_70mm
    gyroY_L_72mm
    gyroY_L_74mm
    gyroY_L_76mm
    gyroY_L_78mm
    gyroY_L_80mm
    gyroY_L_82mm
    gyroY_L_84mm
    gyroY_L_86mm
    gyroY_L_88mm
    gyroY_L_90mm
    gyroY_L_92mm
    gyroY_L_94mm
    gyroY_L_96mm
    gyroY_L_98mm
    gyroY_L_100mm
    gyroY_L_max_mm
    gyroY_L_min_mm
    gyroZ_L_0
    gyroZ_L_2
    gyroZ_L_4
    gyroZ_L_6
    gyroZ_L_8
    gyroZ_L_10
    gyroZ_L_12
    gyroZ_L_14
    gyroZ_L_16
    gyroZ_L_18
    gyroZ_L_20
    gyroZ_L_22
    gyroZ_L_24
    gyroZ_L_26
    gyroZ_L_28
    gyroZ_L_30
    gyroZ_L_32
    gyroZ_L_34
    gyroZ_L_36
    gyroZ_L_38
    gyroZ_L_40
    gyroZ_L_42
    gyroZ_L_44
    gyroZ_L_46
    gyroZ_L_48
    gyroZ_L_50
    gyroZ_L_52
    gyroZ_L_54
    gyroZ_L_56
    gyroZ_L_58
    gyroZ_L_60
    gyroZ_L_62
    gyroZ_L_64
    gyroZ_L_66
    gyroZ_L_68
    gyroZ_L_70
    gyroZ_L_72
    gyroZ_L_74
    gyroZ_L_76
    gyroZ_L_78
    gyroZ_L_80
    gyroZ_L_82
    gyroZ_L_84
    gyroZ_L_86
    gyroZ_L_88
    gyroZ_L_90
    gyroZ_L_92
    gyroZ_L_94
    gyroZ_L_96
    gyroZ_L_98
    gyroZ_L_100
    gyroZ_L_max
    gyroZ_L_min
    gyroZ_L_mean
    gyroZ_L_0mm
    gyroZ_L_2mm
    gyroZ_L_4mm
    gyroZ_L_6mm
    gyroZ_L_8mm
    gyroZ_L_10mm
    gyroZ_L_12mm
    gyroZ_L_14mm
    gyroZ_L_16mm
    gyroZ_L_18mm
    gyroZ_L_20mm
    gyroZ_L_22mm
    gyroZ_L_24mm
    gyroZ_L_26mm
    gyroZ_L_28mm
    gyroZ_L_30mm
    gyroZ_L_32mm
    gyroZ_L_34mm
    gyroZ_L_36mm
    gyroZ_L_38mm
    gyroZ_L_40mm
    gyroZ_L_42mm
    gyroZ_L_44mm
    gyroZ_L_46mm
    gyroZ_L_48mm
    gyroZ_L_50mm
    gyroZ_L_52mm
    gyroZ_L_54mm
    gyroZ_L_56mm
    gyroZ_L_58mm
    gyroZ_L_60mm
    gyroZ_L_62mm
    gyroZ_L_64mm
    gyroZ_L_66mm
    gyroZ_L_68mm
    gyroZ_L_70mm
    gyroZ_L_72mm
    gyroZ_L_74mm
    gyroZ_L_76mm
    gyroZ_L_78mm
    gyroZ_L_80mm
    gyroZ_L_82mm
    gyroZ_L_84mm
    gyroZ_L_86mm
    gyroZ_L_88mm
    gyroZ_L_90mm
    gyroZ_L_92mm
    gyroZ_L_94mm
    gyroZ_L_96mm
    gyroZ_L_98mm
    gyroZ_L_100mm
    gyroZ_L_max_mm
    gyroZ_L_min_mm
    accXRot_L_0
    accXRot_L_2
    accXRot_L_4
    accXRot_L_6
    accXRot_L_8
    accXRot_L_10
    accXRot_L_12
    accXRot_L_14
    accXRot_L_16
    accXRot_L_18
    accXRot_L_20
    accXRot_L_22
    accXRot_L_24
    accXRot_L_26
    accXRot_L_28
    accXRot_L_30
    accXRot_L_32
    accXRot_L_34
    accXRot_L_36
    accXRot_L_38
    accXRot_L_40
    accXRot_L_42
    accXRot_L_44
    accXRot_L_46
    accXRot_L_48
    accXRot_L_50
    accXRot_L_52
    accXRot_L_54
    accXRot_L_56
    accXRot_L_58
    accXRot_L_60
    accXRot_L_62
    accXRot_L_64
    accXRot_L_66
    accXRot_L_68
    accXRot_L_70
    accXRot_L_72
    accXRot_L_74
    accXRot_L_76
    accXRot_L_78
    accXRot_L_80
    accXRot_L_82
    accXRot_L_84
    accXRot_L_86
    accXRot_L_88
    accXRot_L_90
    accXRot_L_92
    accXRot_L_94
    accXRot_L_96
    accXRot_L_98
    accXRot_L_100
    accXRot_L_max
    accXRot_L_min
    accXRot_L_mean
    accXRot_L_0mm
    accXRot_L_2mm
    accXRot_L_4mm
    accXRot_L_6mm
    accXRot_L_8mm
    accXRot_L_10mm
    accXRot_L_12mm
    accXRot_L_14mm
    accXRot_L_16mm
    accXRot_L_18mm
    accXRot_L_20mm
    accXRot_L_22mm
    accXRot_L_24mm
    accXRot_L_26mm
    accXRot_L_28mm
    accXRot_L_30mm
    accXRot_L_32mm
    accXRot_L_34mm
    accXRot_L_36mm
    accXRot_L_38mm
    accXRot_L_40mm
    accXRot_L_42mm
    accXRot_L_44mm
    accXRot_L_46mm
    accXRot_L_48mm
    accXRot_L_50mm
    accXRot_L_52mm
    accXRot_L_54mm
    accXRot_L_56mm
    accXRot_L_58mm
    accXRot_L_60mm
    accXRot_L_62mm
    accXRot_L_64mm
    accXRot_L_66mm
    accXRot_L_68mm
    accXRot_L_70mm
    accXRot_L_72mm
    accXRot_L_74mm
    accXRot_L_76mm
    accXRot_L_78mm
    accXRot_L_80mm
    accXRot_L_82mm
    accXRot_L_84mm
    accXRot_L_86mm
    accXRot_L_88mm
    accXRot_L_90mm
    accXRot_L_92mm
    accXRot_L_94mm
    accXRot_L_96mm
    accXRot_L_98mm
    accXRot_L_100mm
    accXRot_L_max_mm
    accXRot_L_min_mm
    accYRot_L_0
    accYRot_L_2
    accYRot_L_4
    accYRot_L_6
    accYRot_L_8
    accYRot_L_10
    accYRot_L_12
    accYRot_L_14
    accYRot_L_16
    accYRot_L_18
    accYRot_L_20
    accYRot_L_22
    accYRot_L_24
    accYRot_L_26
    accYRot_L_28
    accYRot_L_30
    accYRot_L_32
    accYRot_L_34
    accYRot_L_36
    accYRot_L_38
    accYRot_L_40
    accYRot_L_42
    accYRot_L_44
    accYRot_L_46
    accYRot_L_48
    accYRot_L_50
    accYRot_L_52
    accYRot_L_54
    accYRot_L_56
    accYRot_L_58
    accYRot_L_60
    accYRot_L_62
    accYRot_L_64
    accYRot_L_66
    accYRot_L_68
    accYRot_L_70
    accYRot_L_72
    accYRot_L_74
    accYRot_L_76
    accYRot_L_78
    accYRot_L_80
    accYRot_L_82
    accYRot_L_84
    accYRot_L_86
    accYRot_L_88
    accYRot_L_90
    accYRot_L_92
    accYRot_L_94
    accYRot_L_96
    accYRot_L_98
    accYRot_L_100
    accYRot_L_max
    accYRot_L_min
    accYRot_L_mean
    accYRot_L_0mm
    accYRot_L_2mm
    accYRot_L_4mm
    accYRot_L_6mm
    accYRot_L_8mm
    accYRot_L_10mm
    accYRot_L_12mm
    accYRot_L_14mm
    accYRot_L_16mm
    accYRot_L_18mm
    accYRot_L_20mm
    accYRot_L_22mm
    accYRot_L_24mm
    accYRot_L_26mm
    accYRot_L_28mm
    accYRot_L_30mm
    accYRot_L_32mm
    accYRot_L_34mm
    accYRot_L_36mm
    accYRot_L_38mm
    accYRot_L_40mm
    accYRot_L_42mm
    accYRot_L_44mm
    accYRot_L_46mm
    accYRot_L_48mm
    accYRot_L_50mm
    accYRot_L_52mm
    accYRot_L_54mm
    accYRot_L_56mm
    accYRot_L_58mm
    accYRot_L_60mm
    accYRot_L_62mm
    accYRot_L_64mm
    accYRot_L_66mm
    accYRot_L_68mm
    accYRot_L_70mm
    accYRot_L_72mm
    accYRot_L_74mm
    accYRot_L_76mm
    accYRot_L_78mm
    accYRot_L_80mm
    accYRot_L_82mm
    accYRot_L_84mm
    accYRot_L_86mm
    accYRot_L_88mm
    accYRot_L_90mm
    accYRot_L_92mm
    accYRot_L_94mm
    accYRot_L_96mm
    accYRot_L_98mm
    accYRot_L_100mm
    accYRot_L_max_mm
    accYRot_L_min_mm
    accZRot_L_0
    accZRot_L_2
    accZRot_L_4
    accZRot_L_6
    accZRot_L_8
    accZRot_L_10
    accZRot_L_12
    accZRot_L_14
    accZRot_L_16
    accZRot_L_18
    accZRot_L_20
    accZRot_L_22
    accZRot_L_24
    accZRot_L_26
    accZRot_L_28
    accZRot_L_30
    accZRot_L_32
    accZRot_L_34
    accZRot_L_36
    accZRot_L_38
    accZRot_L_40
    accZRot_L_42
    accZRot_L_44
    accZRot_L_46
    accZRot_L_48
    accZRot_L_50
    accZRot_L_52
    accZRot_L_54
    accZRot_L_56
    accZRot_L_58
    accZRot_L_60
    accZRot_L_62
    accZRot_L_64
    accZRot_L_66
    accZRot_L_68
    accZRot_L_70
    accZRot_L_72
    accZRot_L_74
    accZRot_L_76
    accZRot_L_78
    accZRot_L_80
    accZRot_L_82
    accZRot_L_84
    accZRot_L_86
    accZRot_L_88
    accZRot_L_90
    accZRot_L_92
    accZRot_L_94
    accZRot_L_96
    accZRot_L_98
    accZRot_L_100
    accZRot_L_max
    accZRot_L_min
    accZRot_L_mean
    accZRot_L_0mm
    accZRot_L_2mm
    accZRot_L_4mm
    accZRot_L_6mm
    accZRot_L_8mm
    accZRot_L_10mm
    accZRot_L_12mm
    accZRot_L_14mm
    accZRot_L_16mm
    accZRot_L_18mm
    accZRot_L_20mm
    accZRot_L_22mm
    accZRot_L_24mm
    accZRot_L_26mm
    accZRot_L_28mm
    accZRot_L_30mm
    accZRot_L_32mm
    accZRot_L_34mm
    accZRot_L_36mm
    accZRot_L_38mm
    accZRot_L_40mm
    accZRot_L_42mm
    accZRot_L_44mm
    accZRot_L_46mm
    accZRot_L_48mm
    accZRot_L_50mm
    accZRot_L_52mm
    accZRot_L_54mm
    accZRot_L_56mm
    accZRot_L_58mm
    accZRot_L_60mm
    accZRot_L_62mm
    accZRot_L_64mm
    accZRot_L_66mm
    accZRot_L_68mm
    accZRot_L_70mm
    accZRot_L_72mm
    accZRot_L_74mm
    accZRot_L_76mm
    accZRot_L_78mm
    accZRot_L_80mm
    accZRot_L_82mm
    accZRot_L_84mm
    accZRot_L_86mm
    accZRot_L_88mm
    accZRot_L_90mm
    accZRot_L_92mm
    accZRot_L_94mm
    accZRot_L_96mm
    accZRot_L_98mm
    accZRot_L_100mm
    accZRot_L_max_mm
    accZRot_L_min_mm
    angleX_A_0
    angleX_A_2
    angleX_A_4
    angleX_A_6
    angleX_A_8
    angleX_A_10
    angleX_A_12
    angleX_A_14
    angleX_A_16
    angleX_A_18
    angleX_A_20
    angleX_A_22
    angleX_A_24
    angleX_A_26
    angleX_A_28
    angleX_A_30
    angleX_A_32
    angleX_A_34
    angleX_A_36
    angleX_A_38
    angleX_A_40
    angleX_A_42
    angleX_A_44
    angleX_A_46
    angleX_A_48
    angleX_A_50
    angleX_A_52
    angleX_A_54
    angleX_A_56
    angleX_A_58
    angleX_A_60
    angleX_A_62
    angleX_A_64
    angleX_A_66
    angleX_A_68
    angleX_A_70
    angleX_A_72
    angleX_A_74
    angleX_A_76
    angleX_A_78
    angleX_A_80
    angleX_A_82
    angleX_A_84
    angleX_A_86
    angleX_A_88
    angleX_A_90
    angleX_A_92
    angleX_A_94
    angleX_A_96
    angleX_A_98
    angleX_A_100
    angleX_A_max
    angleX_A_min
    angleX_A_mean
    angleX_A_0mm
    angleX_A_2mm
    angleX_A_4mm
    angleX_A_6mm
    angleX_A_8mm
    angleX_A_10mm
    angleX_A_12mm
    angleX_A_14mm
    angleX_A_16mm
    angleX_A_18mm
    angleX_A_20mm
    angleX_A_22mm
    angleX_A_24mm
    angleX_A_26mm
    angleX_A_28mm
    angleX_A_30mm
    angleX_A_32mm
    angleX_A_34mm
    angleX_A_36mm
    angleX_A_38mm
    angleX_A_40mm
    angleX_A_42mm
    angleX_A_44mm
    angleX_A_46mm
    angleX_A_48mm
    angleX_A_50mm
    angleX_A_52mm
    angleX_A_54mm
    angleX_A_56mm
    angleX_A_58mm
    angleX_A_60mm
    angleX_A_62mm
    angleX_A_64mm
    angleX_A_66mm
    angleX_A_68mm
    angleX_A_70mm
    angleX_A_72mm
    angleX_A_74mm
    angleX_A_76mm
    angleX_A_78mm
    angleX_A_80mm
    angleX_A_82mm
    angleX_A_84mm
    angleX_A_86mm
    angleX_A_88mm
    angleX_A_90mm
    angleX_A_92mm
    angleX_A_94mm
    angleX_A_96mm
    angleX_A_98mm
    angleX_A_100mm
    angleX_A_max_mm
    angleX_A_min_mm
    angleY_A_0
    angleY_A_2
    angleY_A_4
    angleY_A_6
    angleY_A_8
    angleY_A_10
    angleY_A_12
    angleY_A_14
    angleY_A_16
    angleY_A_18
    angleY_A_20
    angleY_A_22
    angleY_A_24
    angleY_A_26
    angleY_A_28
    angleY_A_30
    angleY_A_32
    angleY_A_34
    angleY_A_36
    angleY_A_38
    angleY_A_40
    angleY_A_42
    angleY_A_44
    angleY_A_46
    angleY_A_48
    angleY_A_50
    angleY_A_52
    angleY_A_54
    angleY_A_56
    angleY_A_58
    angleY_A_60
    angleY_A_62
    angleY_A_64
    angleY_A_66
    angleY_A_68
    angleY_A_70
    angleY_A_72
    angleY_A_74
    angleY_A_76
    angleY_A_78
    angleY_A_80
    angleY_A_82
    angleY_A_84
    angleY_A_86
    angleY_A_88
    angleY_A_90
    angleY_A_92
    angleY_A_94
    angleY_A_96
    angleY_A_98
    angleY_A_100
    angleY_A_max
    angleY_A_min
    angleY_A_mean
    angleY_A_0mm
    angleY_A_2mm
    angleY_A_4mm
    angleY_A_6mm
    angleY_A_8mm
    angleY_A_10mm
    angleY_A_12mm
    angleY_A_14mm
    angleY_A_16mm
    angleY_A_18mm
    angleY_A_20mm
    angleY_A_22mm
    angleY_A_24mm
    angleY_A_26mm
    angleY_A_28mm
    angleY_A_30mm
    angleY_A_32mm
    angleY_A_34mm
    angleY_A_36mm
    angleY_A_38mm
    angleY_A_40mm
    angleY_A_42mm
    angleY_A_44mm
    angleY_A_46mm
    angleY_A_48mm
    angleY_A_50mm
    angleY_A_52mm
    angleY_A_54mm
    angleY_A_56mm
    angleY_A_58mm
    angleY_A_60mm
    angleY_A_62mm
    angleY_A_64mm
    angleY_A_66mm
    angleY_A_68mm
    angleY_A_70mm
    angleY_A_72mm
    angleY_A_74mm
    angleY_A_76mm
    angleY_A_78mm
    angleY_A_80mm
    angleY_A_82mm
    angleY_A_84mm
    angleY_A_86mm
    angleY_A_88mm
    angleY_A_90mm
    angleY_A_92mm
    angleY_A_94mm
    angleY_A_96mm
    angleY_A_98mm
    angleY_A_100mm
    angleY_A_max_mm
    angleY_A_min_mm
    angleZ_A_0
    angleZ_A_2
    angleZ_A_4
    angleZ_A_6
    angleZ_A_8
    angleZ_A_10
    angleZ_A_12
    angleZ_A_14
    angleZ_A_16
    angleZ_A_18
    angleZ_A_20
    angleZ_A_22
    angleZ_A_24
    angleZ_A_26
    angleZ_A_28
    angleZ_A_30
    angleZ_A_32
    angleZ_A_34
    angleZ_A_36
    angleZ_A_38
    angleZ_A_40
    angleZ_A_42
    angleZ_A_44
    angleZ_A_46
    angleZ_A_48
    angleZ_A_50
    angleZ_A_52
    angleZ_A_54
    angleZ_A_56
    angleZ_A_58
    angleZ_A_60
    angleZ_A_62
    angleZ_A_64
    angleZ_A_66
    angleZ_A_68
    angleZ_A_70
    angleZ_A_72
    angleZ_A_74
    angleZ_A_76
    angleZ_A_78
    angleZ_A_80
    angleZ_A_82
    angleZ_A_84
    angleZ_A_86
    angleZ_A_88
    angleZ_A_90
    angleZ_A_92
    angleZ_A_94
    angleZ_A_96
    angleZ_A_98
    angleZ_A_100
    angleZ_A_max
    angleZ_A_min
    angleZ_A_mean
    angleZ_A_0mm
    angleZ_A_2mm
    angleZ_A_4mm
    angleZ_A_6mm
    angleZ_A_8mm
    angleZ_A_10mm
    angleZ_A_12mm
    angleZ_A_14mm
    angleZ_A_16mm
    angleZ_A_18mm
    angleZ_A_20mm
    angleZ_A_22mm
    angleZ_A_24mm
    angleZ_A_26mm
    angleZ_A_28mm
    angleZ_A_30mm
    angleZ_A_32mm
    angleZ_A_34mm
    angleZ_A_36mm
    angleZ_A_38mm
    angleZ_A_40mm
    angleZ_A_42mm
    angleZ_A_44mm
    angleZ_A_46mm
    angleZ_A_48mm
    angleZ_A_50mm
    angleZ_A_52mm
    angleZ_A_54mm
    angleZ_A_56mm
    angleZ_A_58mm
    angleZ_A_60mm
    angleZ_A_62mm
    angleZ_A_64mm
    angleZ_A_66mm
    angleZ_A_68mm
    angleZ_A_70mm
    angleZ_A_72mm
    angleZ_A_74mm
    angleZ_A_76mm
    angleZ_A_78mm
    angleZ_A_80mm
    angleZ_A_82mm
    angleZ_A_84mm
    angleZ_A_86mm
    angleZ_A_88mm
    angleZ_A_90mm
    angleZ_A_92mm
    angleZ_A_94mm
    angleZ_A_96mm
    angleZ_A_98mm
    angleZ_A_100mm
    angleZ_A_max_mm
    angleZ_A_min_mm
    gyroX_A_0
    gyroX_A_2
    gyroX_A_4
    gyroX_A_6
    gyroX_A_8
    gyroX_A_10
    gyroX_A_12
    gyroX_A_14
    gyroX_A_16
    gyroX_A_18
    gyroX_A_20
    gyroX_A_22
    gyroX_A_24
    gyroX_A_26
    gyroX_A_28
    gyroX_A_30
    gyroX_A_32
    gyroX_A_34
    gyroX_A_36
    gyroX_A_38
    gyroX_A_40
    gyroX_A_42
    gyroX_A_44
    gyroX_A_46
    gyroX_A_48
    gyroX_A_50
    gyroX_A_52
    gyroX_A_54
    gyroX_A_56
    gyroX_A_58
    gyroX_A_60
    gyroX_A_62
    gyroX_A_64
    gyroX_A_66
    gyroX_A_68
    gyroX_A_70
    gyroX_A_72
    gyroX_A_74
    gyroX_A_76
    gyroX_A_78
    gyroX_A_80
    gyroX_A_82
    gyroX_A_84
    gyroX_A_86
    gyroX_A_88
    gyroX_A_90
    gyroX_A_92
    gyroX_A_94
    gyroX_A_96
    gyroX_A_98
    gyroX_A_100
    gyroX_A_max
    gyroX_A_min
    gyroX_A_mean
    gyroX_A_0mm
    gyroX_A_2mm
    gyroX_A_4mm
    gyroX_A_6mm
    gyroX_A_8mm
    gyroX_A_10mm
    gyroX_A_12mm
    gyroX_A_14mm
    gyroX_A_16mm
    gyroX_A_18mm
    gyroX_A_20mm
    gyroX_A_22mm
    gyroX_A_24mm
    gyroX_A_26mm
    gyroX_A_28mm
    gyroX_A_30mm
    gyroX_A_32mm
    gyroX_A_34mm
    gyroX_A_36mm
    gyroX_A_38mm
    gyroX_A_40mm
    gyroX_A_42mm
    gyroX_A_44mm
    gyroX_A_46mm
    gyroX_A_48mm
    gyroX_A_50mm
    gyroX_A_52mm
    gyroX_A_54mm
    gyroX_A_56mm
    gyroX_A_58mm
    gyroX_A_60mm
    gyroX_A_62mm
    gyroX_A_64mm
    gyroX_A_66mm
    gyroX_A_68mm
    gyroX_A_70mm
    gyroX_A_72mm
    gyroX_A_74mm
    gyroX_A_76mm
    gyroX_A_78mm
    gyroX_A_80mm
    gyroX_A_82mm
    gyroX_A_84mm
    gyroX_A_86mm
    gyroX_A_88mm
    gyroX_A_90mm
    gyroX_A_92mm
    gyroX_A_94mm
    gyroX_A_96mm
    gyroX_A_98mm
    gyroX_A_100mm
    gyroX_A_max_mm
    gyroX_A_min_mm
    gyroY_A_0
    gyroY_A_2
    gyroY_A_4
    gyroY_A_6
    gyroY_A_8
    gyroY_A_10
    gyroY_A_12
    gyroY_A_14
    gyroY_A_16
    gyroY_A_18
    gyroY_A_20
    gyroY_A_22
    gyroY_A_24
    gyroY_A_26
    gyroY_A_28
    gyroY_A_30
    gyroY_A_32
    gyroY_A_34
    gyroY_A_36
    gyroY_A_38
    gyroY_A_40
    gyroY_A_42
    gyroY_A_44
    gyroY_A_46
    gyroY_A_48
    gyroY_A_50
    gyroY_A_52
    gyroY_A_54
    gyroY_A_56
    gyroY_A_58
    gyroY_A_60
    gyroY_A_62
    gyroY_A_64
    gyroY_A_66
    gyroY_A_68
    gyroY_A_70
    gyroY_A_72
    gyroY_A_74
    gyroY_A_76
    gyroY_A_78
    gyroY_A_80
    gyroY_A_82
    gyroY_A_84
    gyroY_A_86
    gyroY_A_88
    gyroY_A_90
    gyroY_A_92
    gyroY_A_94
    gyroY_A_96
    gyroY_A_98
    gyroY_A_100
    gyroY_A_max
    gyroY_A_min
    gyroY_A_mean
    gyroY_A_0mm
    gyroY_A_2mm
    gyroY_A_4mm
    gyroY_A_6mm
    gyroY_A_8mm
    gyroY_A_10mm
    gyroY_A_12mm
    gyroY_A_14mm
    gyroY_A_16mm
    gyroY_A_18mm
    gyroY_A_20mm
    gyroY_A_22mm
    gyroY_A_24mm
    gyroY_A_26mm
    gyroY_A_28mm
    gyroY_A_30mm
    gyroY_A_32mm
    gyroY_A_34mm
    gyroY_A_36mm
    gyroY_A_38mm
    gyroY_A_40mm
    gyroY_A_42mm
    gyroY_A_44mm
    gyroY_A_46mm
    gyroY_A_48mm
    gyroY_A_50mm
    gyroY_A_52mm
    gyroY_A_54mm
    gyroY_A_56mm
    gyroY_A_58mm
    gyroY_A_60mm
    gyroY_A_62mm
    gyroY_A_64mm
    gyroY_A_66mm
    gyroY_A_68mm
    gyroY_A_70mm
    gyroY_A_72mm
    gyroY_A_74mm
    gyroY_A_76mm
    gyroY_A_78mm
    gyroY_A_80mm
    gyroY_A_82mm
    gyroY_A_84mm
    gyroY_A_86mm
    gyroY_A_88mm
    gyroY_A_90mm
    gyroY_A_92mm
    gyroY_A_94mm
    gyroY_A_96mm
    gyroY_A_98mm
    gyroY_A_100mm
    gyroY_A_max_mm
    gyroY_A_min_mm
    gyroZ_A_0
    gyroZ_A_2
    gyroZ_A_4
    gyroZ_A_6
    gyroZ_A_8
    gyroZ_A_10
    gyroZ_A_12
    gyroZ_A_14
    gyroZ_A_16
    gyroZ_A_18
    gyroZ_A_20
    gyroZ_A_22
    gyroZ_A_24
    gyroZ_A_26
    gyroZ_A_28
    gyroZ_A_30
    gyroZ_A_32
    gyroZ_A_34
    gyroZ_A_36
    gyroZ_A_38
    gyroZ_A_40
    gyroZ_A_42
    gyroZ_A_44
    gyroZ_A_46
    gyroZ_A_48
    gyroZ_A_50
    gyroZ_A_52
    gyroZ_A_54
    gyroZ_A_56
    gyroZ_A_58
    gyroZ_A_60
    gyroZ_A_62
    gyroZ_A_64
    gyroZ_A_66
    gyroZ_A_68
    gyroZ_A_70
    gyroZ_A_72
    gyroZ_A_74
    gyroZ_A_76
    gyroZ_A_78
    gyroZ_A_80
    gyroZ_A_82
    gyroZ_A_84
    gyroZ_A_86
    gyroZ_A_88
    gyroZ_A_90
    gyroZ_A_92
    gyroZ_A_94
    gyroZ_A_96
    gyroZ_A_98
    gyroZ_A_100
    gyroZ_A_max
    gyroZ_A_min
    gyroZ_A_mean
    gyroZ_A_0mm
    gyroZ_A_2mm
    gyroZ_A_4mm
    gyroZ_A_6mm
    gyroZ_A_8mm
    gyroZ_A_10mm
    gyroZ_A_12mm
    gyroZ_A_14mm
    gyroZ_A_16mm
    gyroZ_A_18mm
    gyroZ_A_20mm
    gyroZ_A_22mm
    gyroZ_A_24mm
    gyroZ_A_26mm
    gyroZ_A_28mm
    gyroZ_A_30mm
    gyroZ_A_32mm
    gyroZ_A_34mm
    gyroZ_A_36mm
    gyroZ_A_38mm
    gyroZ_A_40mm
    gyroZ_A_42mm
    gyroZ_A_44mm
    gyroZ_A_46mm
    gyroZ_A_48mm
    gyroZ_A_50mm
    gyroZ_A_52mm
    gyroZ_A_54mm
    gyroZ_A_56mm
    gyroZ_A_58mm
    gyroZ_A_60mm
    gyroZ_A_62mm
    gyroZ_A_64mm
    gyroZ_A_66mm
    gyroZ_A_68mm
    gyroZ_A_70mm
    gyroZ_A_72mm
    gyroZ_A_74mm
    gyroZ_A_76mm
    gyroZ_A_78mm
    gyroZ_A_80mm
    gyroZ_A_82mm
    gyroZ_A_84mm
    gyroZ_A_86mm
    gyroZ_A_88mm
    gyroZ_A_90mm
    gyroZ_A_92mm
    gyroZ_A_94mm
    gyroZ_A_96mm
    gyroZ_A_98mm
    gyroZ_A_100mm
    gyroZ_A_max_mm
    gyroZ_A_min_mm
    accXRot_A_0
    accXRot_A_2
    accXRot_A_4
    accXRot_A_6
    accXRot_A_8
    accXRot_A_10
    accXRot_A_12
    accXRot_A_14
    accXRot_A_16
    accXRot_A_18
    accXRot_A_20
    accXRot_A_22
    accXRot_A_24
    accXRot_A_26
    accXRot_A_28
    accXRot_A_30
    accXRot_A_32
    accXRot_A_34
    accXRot_A_36
    accXRot_A_38
    accXRot_A_40
    accXRot_A_42
    accXRot_A_44
    accXRot_A_46
    accXRot_A_48
    accXRot_A_50
    accXRot_A_52
    accXRot_A_54
    accXRot_A_56
    accXRot_A_58
    accXRot_A_60
    accXRot_A_62
    accXRot_A_64
    accXRot_A_66
    accXRot_A_68
    accXRot_A_70
    accXRot_A_72
    accXRot_A_74
    accXRot_A_76
    accXRot_A_78
    accXRot_A_80
    accXRot_A_82
    accXRot_A_84
    accXRot_A_86
    accXRot_A_88
    accXRot_A_90
    accXRot_A_92
    accXRot_A_94
    accXRot_A_96
    accXRot_A_98
    accXRot_A_100
    accXRot_A_max
    accXRot_A_min
    accXRot_A_mean
    accXRot_A_0mm
    accXRot_A_2mm
    accXRot_A_4mm
    accXRot_A_6mm
    accXRot_A_8mm
    accXRot_A_10mm
    accXRot_A_12mm
    accXRot_A_14mm
    accXRot_A_16mm
    accXRot_A_18mm
    accXRot_A_20mm
    accXRot_A_22mm
    accXRot_A_24mm
    accXRot_A_26mm
    accXRot_A_28mm
    accXRot_A_30mm
    accXRot_A_32mm
    accXRot_A_34mm
    accXRot_A_36mm
    accXRot_A_38mm
    accXRot_A_40mm
    accXRot_A_42mm
    accXRot_A_44mm
    accXRot_A_46mm
    accXRot_A_48mm
    accXRot_A_50mm
    accXRot_A_52mm
    accXRot_A_54mm
    accXRot_A_56mm
    accXRot_A_58mm
    accXRot_A_60mm
    accXRot_A_62mm
    accXRot_A_64mm
    accXRot_A_66mm
    accXRot_A_68mm
    accXRot_A_70mm
    accXRot_A_72mm
    accXRot_A_74mm
    accXRot_A_76mm
    accXRot_A_78mm
    accXRot_A_80mm
    accXRot_A_82mm
    accXRot_A_84mm
    accXRot_A_86mm
    accXRot_A_88mm
    accXRot_A_90mm
    accXRot_A_92mm
    accXRot_A_94mm
    accXRot_A_96mm
    accXRot_A_98mm
    accXRot_A_100mm
    accXRot_A_max_mm
    accXRot_A_min_mm
    accYRot_A_0
    accYRot_A_2
    accYRot_A_4
    accYRot_A_6
    accYRot_A_8
    accYRot_A_10
    accYRot_A_12
    accYRot_A_14
    accYRot_A_16
    accYRot_A_18
    accYRot_A_20
    accYRot_A_22
    accYRot_A_24
    accYRot_A_26
    accYRot_A_28
    accYRot_A_30
    accYRot_A_32
    accYRot_A_34
    accYRot_A_36
    accYRot_A_38
    accYRot_A_40
    accYRot_A_42
    accYRot_A_44
    accYRot_A_46
    accYRot_A_48
    accYRot_A_50
    accYRot_A_52
    accYRot_A_54
    accYRot_A_56
    accYRot_A_58
    accYRot_A_60
    accYRot_A_62
    accYRot_A_64
    accYRot_A_66
    accYRot_A_68
    accYRot_A_70
    accYRot_A_72
    accYRot_A_74
    accYRot_A_76
    accYRot_A_78
    accYRot_A_80
    accYRot_A_82
    accYRot_A_84
    accYRot_A_86
    accYRot_A_88
    accYRot_A_90
    accYRot_A_92
    accYRot_A_94
    accYRot_A_96
    accYRot_A_98
    accYRot_A_100
    accYRot_A_max
    accYRot_A_min
    accYRot_A_mean
    accYRot_A_0mm
    accYRot_A_2mm
    accYRot_A_4mm
    accYRot_A_6mm
    accYRot_A_8mm
    accYRot_A_10mm
    accYRot_A_12mm
    accYRot_A_14mm
    accYRot_A_16mm
    accYRot_A_18mm
    accYRot_A_20mm
    accYRot_A_22mm
    accYRot_A_24mm
    accYRot_A_26mm
    accYRot_A_28mm
    accYRot_A_30mm
    accYRot_A_32mm
    accYRot_A_34mm
    accYRot_A_36mm
    accYRot_A_38mm
    accYRot_A_40mm
    accYRot_A_42mm
    accYRot_A_44mm
    accYRot_A_46mm
    accYRot_A_48mm
    accYRot_A_50mm
    accYRot_A_52mm
    accYRot_A_54mm
    accYRot_A_56mm
    accYRot_A_58mm
    accYRot_A_60mm
    accYRot_A_62mm
    accYRot_A_64mm
    accYRot_A_66mm
    accYRot_A_68mm
    accYRot_A_70mm
    accYRot_A_72mm
    accYRot_A_74mm
    accYRot_A_76mm
    accYRot_A_78mm
    accYRot_A_80mm
    accYRot_A_82mm
    accYRot_A_84mm
    accYRot_A_86mm
    accYRot_A_88mm
    accYRot_A_90mm
    accYRot_A_92mm
    accYRot_A_94mm
    accYRot_A_96mm
    accYRot_A_98mm
    accYRot_A_100mm
    accYRot_A_max_mm
    accYRot_A_min_mm
    accZRot_A_0
    accZRot_A_2
    accZRot_A_4
    accZRot_A_6
    accZRot_A_8
    accZRot_A_10
    accZRot_A_12
    accZRot_A_14
    accZRot_A_16
    accZRot_A_18
    accZRot_A_20
    accZRot_A_22
    accZRot_A_24
    accZRot_A_26
    accZRot_A_28
    accZRot_A_30
    accZRot_A_32
    accZRot_A_34
    accZRot_A_36
    accZRot_A_38
    accZRot_A_40
    accZRot_A_42
    accZRot_A_44
    accZRot_A_46
    accZRot_A_48
    accZRot_A_50
    accZRot_A_52
    accZRot_A_54
    accZRot_A_56
    accZRot_A_58
    accZRot_A_60
    accZRot_A_62
    accZRot_A_64
    accZRot_A_66
    accZRot_A_68
    accZRot_A_70
    accZRot_A_72
    accZRot_A_74
    accZRot_A_76
    accZRot_A_78
    accZRot_A_80
    accZRot_A_82
    accZRot_A_84
    accZRot_A_86
    accZRot_A_88
    accZRot_A_90
    accZRot_A_92
    accZRot_A_94
    accZRot_A_96
    accZRot_A_98
    accZRot_A_100
    accZRot_A_max
    accZRot_A_min
    accZRot_A_mean
    accZRot_A_0mm
    accZRot_A_2mm
    accZRot_A_4mm
    accZRot_A_6mm
    accZRot_A_8mm
    accZRot_A_10mm
    accZRot_A_12mm
    accZRot_A_14mm
    accZRot_A_16mm
    accZRot_A_18mm
    accZRot_A_20mm
    accZRot_A_22mm
    accZRot_A_24mm
    accZRot_A_26mm
    accZRot_A_28mm
    accZRot_A_30mm
    accZRot_A_32mm
    accZRot_A_34mm
    accZRot_A_36mm
    accZRot_A_38mm
    accZRot_A_40mm
    accZRot_A_42mm
    accZRot_A_44mm
    accZRot_A_46mm
    accZRot_A_48mm
    accZRot_A_50mm
    accZRot_A_52mm
    accZRot_A_54mm
    accZRot_A_56mm
    accZRot_A_58mm
    accZRot_A_60mm
    accZRot_A_62mm
    accZRot_A_64mm
    accZRot_A_66mm
    accZRot_A_68mm
    accZRot_A_70mm
    accZRot_A_72mm
    accZRot_A_74mm
    accZRot_A_76mm
    accZRot_A_78mm
    accZRot_A_80mm
    accZRot_A_82mm
    accZRot_A_84mm
    accZRot_A_86mm
    accZRot_A_88mm
    accZRot_A_90mm
    accZRot_A_92mm
    accZRot_A_94mm
    accZRot_A_96mm
    accZRot_A_98mm
    accZRot_A_100mm
    accZRot_A_max_mm
    accZRot_A_min_mm
    timeGaitExt_L_0
    timeGaitExt_L_2
    timeGaitExt_L_4
    timeGaitExt_L_6
    timeGaitExt_L_8
    timeGaitExt_L_10
    timeGaitExt_L_12
    timeGaitExt_L_14
    timeGaitExt_L_16
    timeGaitExt_L_18
    timeGaitExt_L_20
    timeGaitExt_L_22
    timeGaitExt_L_24
    timeGaitExt_L_26
    timeGaitExt_L_28
    timeGaitExt_L_30
    timeGaitExt_L_32
    timeGaitExt_L_34
    timeGaitExt_L_36
    timeGaitExt_L_38
    timeGaitExt_L_40
    timeGaitExt_L_42
    timeGaitExt_L_44
    timeGaitExt_L_46
    timeGaitExt_L_48
    timeGaitExt_L_50
    timeGaitExt_L_52
    timeGaitExt_L_54
    timeGaitExt_L_56
    timeGaitExt_L_58
    timeGaitExt_L_60
    timeGaitExt_L_62
    timeGaitExt_L_64
    timeGaitExt_L_66
    timeGaitExt_L_68
    timeGaitExt_L_70
    timeGaitExt_L_72
    timeGaitExt_L_74
    timeGaitExt_L_76
    timeGaitExt_L_78
    timeGaitExt_L_80
    timeGaitExt_L_82
    timeGaitExt_L_84
    timeGaitExt_L_86
    timeGaitExt_L_88
    timeGaitExt_L_90
    timeGaitExt_L_92
    timeGaitExt_L_94
    timeGaitExt_L_96
    timeGaitExt_L_98
    timeGaitExt_L_100
    timeGaitExt_R_0
    timeGaitExt_R_2
    timeGaitExt_R_4
    timeGaitExt_R_6
    timeGaitExt_R_8
    timeGaitExt_R_10
    timeGaitExt_R_12
    timeGaitExt_R_14
    timeGaitExt_R_16
    timeGaitExt_R_18
    timeGaitExt_R_20
    timeGaitExt_R_22
    timeGaitExt_R_24
    timeGaitExt_R_26
    timeGaitExt_R_28
    timeGaitExt_R_30
    timeGaitExt_R_32
    timeGaitExt_R_34
    timeGaitExt_R_36
    timeGaitExt_R_38
    timeGaitExt_R_40
    timeGaitExt_R_42
    timeGaitExt_R_44
    timeGaitExt_R_46
    timeGaitExt_R_48
    timeGaitExt_R_50
    timeGaitExt_R_52
    timeGaitExt_R_54
    timeGaitExt_R_56
    timeGaitExt_R_58
    timeGaitExt_R_60
    timeGaitExt_R_62
    timeGaitExt_R_64
    timeGaitExt_R_66
    timeGaitExt_R_68
    timeGaitExt_R_70
    timeGaitExt_R_72
    timeGaitExt_R_74
    timeGaitExt_R_76
    timeGaitExt_R_78
    timeGaitExt_R_80
    timeGaitExt_R_82
    timeGaitExt_R_84
    timeGaitExt_R_86
    timeGaitExt_R_88
    timeGaitExt_R_90
    timeGaitExt_R_92
    timeGaitExt_R_94
    timeGaitExt_R_96
    timeGaitExt_R_98
    timeGaitExt_R_100
    isrunning_0
    isrunning_2
    isrunning_4
    isrunning_6
    isrunning_8
    isrunning_10
    isrunning_12
    isrunning_14
    isrunning_16
    isrunning_18
    isrunning_20
    isrunning_22
    isrunning_24
    isrunning_26
    isrunning_28
    isrunning_30
    isrunning_32
    isrunning_34
    isrunning_36
    isrunning_38
    isrunning_40
    isrunning_42
    isrunning_44
    isrunning_46
    isrunning_48
    isrunning_50
    isrunning_52
    isrunning_54
    isrunning_56
    isrunning_58
    isrunning_60
    isrunning_62
    isrunning_64
    isrunning_66
    isrunning_68
    isrunning_70
    isrunning_72
    isrunning_74
    isrunning_76
    isrunning_78
    isrunning_80
    isrunning_82
    isrunning_84
    isrunning_86
    isrunning_88
    isrunning_90
    isrunning_92
    isrunning_94
    isrunning_96
    isrunning_98
    isrunning_100
    feature_L_A_0
    feature_L_A_2
    feature_L_A_4
    feature_L_A_6
    feature_L_A_8
    feature_L_A_10
    feature_L_A_12
    feature_L_A_14
    feature_L_A_16
    feature_L_A_18
    feature_L_A_20
    feature_L_A_22
    feature_L_A_24
    feature_L_A_26
    feature_L_A_28
    feature_L_A_30
    feature_L_A_32
    feature_L_A_34
    feature_L_A_36
    feature_L_A_38
    feature_L_A_40
    feature_L_A_42
    feature_L_A_44
    feature_L_A_46
    feature_L_A_48
    feature_L_A_50
    feature_L_A_52
    feature_L_A_54
    feature_L_A_56
    feature_L_A_58
    feature_L_A_60
    feature_L_A_62
    feature_L_A_64
    feature_L_A_66
    feature_L_A_68
    feature_L_A_70
    feature_L_A_72
    feature_L_A_74
    feature_L_A_76
    feature_L_A_78
    feature_L_A_80
    feature_L_A_82
    feature_L_A_84
    feature_L_A_86
    feature_L_A_88
    feature_L_A_90
    feature_L_A_92
    feature_L_A_94
    feature_L_A_96
    feature_L_A_98
    feature_L_A_100
    feature_R_A_0
    feature_R_A_2
    feature_R_A_4
    feature_R_A_6
    feature_R_A_8
    feature_R_A_10
    feature_R_A_12
    feature_R_A_14
    feature_R_A_16
    feature_R_A_18
    feature_R_A_20
    feature_R_A_22
    feature_R_A_24
    feature_R_A_26
    feature_R_A_28
    feature_R_A_30
    feature_R_A_32
    feature_R_A_34
    feature_R_A_36
    feature_R_A_38
    feature_R_A_40
    feature_R_A_42
    feature_R_A_44
    feature_R_A_46
    feature_R_A_48
    feature_R_A_50
    feature_R_A_52
    feature_R_A_54
    feature_R_A_56
    feature_R_A_58
    feature_R_A_60
    feature_R_A_62
    feature_R_A_64
    feature_R_A_66
    feature_R_A_68
    feature_R_A_70
    feature_R_A_72
    feature_R_A_74
    feature_R_A_76
    feature_R_A_78
    feature_R_A_80
    feature_R_A_82
    feature_R_A_84
    feature_R_A_86
    feature_R_A_88
    feature_R_A_90
    feature_R_A_92
    feature_R_A_94
    feature_R_A_96
    feature_R_A_98
    feature_R_A_100
    feature_L_H_0
    feature_L_H_2
    feature_L_H_4
    feature_L_H_6
    feature_L_H_8
    feature_L_H_10
    feature_L_H_12
    feature_L_H_14
    feature_L_H_16
    feature_L_H_18
    feature_L_H_20
    feature_L_H_22
    feature_L_H_24
    feature_L_H_26
    feature_L_H_28
    feature_L_H_30
    feature_L_H_32
    feature_L_H_34
    feature_L_H_36
    feature_L_H_38
    feature_L_H_40
    feature_L_H_42
    feature_L_H_44
    feature_L_H_46
    feature_L_H_48
    feature_L_H_50
    feature_L_H_52
    feature_L_H_54
    feature_L_H_56
    feature_L_H_58
    feature_L_H_60
    feature_L_H_62
    feature_L_H_64
    feature_L_H_66
    feature_L_H_68
    feature_L_H_70
    feature_L_H_72
    feature_L_H_74
    feature_L_H_76
    feature_L_H_78
    feature_L_H_80
    feature_L_H_82
    feature_L_H_84
    feature_L_H_86
    feature_L_H_88
    feature_L_H_90
    feature_L_H_92
    feature_L_H_94
    feature_L_H_96
    feature_L_H_98
    feature_L_H_100
    subHeight
    subWeight
    ref_slope
    slope_class




```python
df_flat.columns[1]
```





    'time_0'





```python
df['ref_speed']
```



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /Users/kate_zym/anaconda/lib/python3.6/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2392             try:
    -> 2393                 return self._engine.get_loc(key)
       2394             except KeyError:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5239)()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5085)()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20405)()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20359)()


    KeyError: 'ref_speed'

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-38-119cbe1dc6fb> in <module>()
    ----> 1 df['ref_speed']
    

    /Users/kate_zym/anaconda/lib/python3.6/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2060             return self._getitem_multilevel(key)
       2061         else:
    -> 2062             return self._getitem_column(key)
       2063 
       2064     def _getitem_column(self, key):


    /Users/kate_zym/anaconda/lib/python3.6/site-packages/pandas/core/frame.py in _getitem_column(self, key)
       2067         # get column
       2068         if self.columns.is_unique:
    -> 2069             return self._get_item_cache(key)
       2070 
       2071         # duplicate columns & possible reduce dimensionality


    /Users/kate_zym/anaconda/lib/python3.6/site-packages/pandas/core/generic.py in _get_item_cache(self, item)
       1532         res = cache.get(item)
       1533         if res is None:
    -> 1534             values = self._data.get(item)
       1535             res = self._box_item_values(item, values)
       1536             cache[item] = res


    /Users/kate_zym/anaconda/lib/python3.6/site-packages/pandas/core/internals.py in get(self, item, fastpath)
       3588 
       3589             if not isnull(item):
    -> 3590                 loc = self.items.get_loc(item)
       3591             else:
       3592                 indexer = np.arange(len(self.items))[isnull(self.items)]


    /Users/kate_zym/anaconda/lib/python3.6/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2393                 return self._engine.get_loc(key)
       2394             except KeyError:
    -> 2395                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2396 
       2397         indexer = self.get_indexer([key], method=method, tolerance=tolerance)


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5239)()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5085)()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20405)()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20359)()


    KeyError: 'ref_speed'




```python

```

