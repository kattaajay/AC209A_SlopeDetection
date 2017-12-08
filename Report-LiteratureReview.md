---
title: Literature Review and Related Work
notebook: Report-LiteratureReview.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}


Before exploring with different modeling methods on the sloped walking Hip-Only system data, we reviewed relevant literature in this field. In general, many different classification models have been succesfully adapted to this field, and below are a few highlights.

### 1. Support vector machine for classification of walking conditions using miniature kinematic sensors
Hong-yin Lau, Kai-yu Tong, Hailong Zhu, Support vector machine for classification of walking conditions of persons after stroke with dropped foot, In Human Movement Science, Volume 28, Issue 4, 2009, Pages 504-514, ISSN 0167-9457, https://doi.org/10.1016/j.humov.2008.12.003.

In this work, shank and foot accelerometer and gyroscope were used to classify walking into 5 walking conditions: uphill, downhill, stair ascent, stair descent, and level ground walking. The data is assembled from 10 gait cycles of each walking condition collected at speeds ranging from 1.4 m/s to 1.47 m/s. Particuarly, preswing phase signals were used to make the classifications. The data was pre-processed using a butterworht filter and maximums and minimums of each sensor signal were extracted as separate predictors. 

The authors built a support vector classifier which out performed other popular methods, such as artificial neural network and Baysian belief network. A SVM classifier was chosen because it is flexible and can adjust the balance between variance and bias by adjust the margin. Using RBF kernel also allowed data to be mapped on to higher dimensional space. The optimal hyperparameters were chosen using kFold cross validation.

In conclusion, the stair ascent and descent classification accuracies were 100%, and the authors noted that adding foot gyros improved the classification rate from 78% to 84%.

### 2. Can Triaxial Accelerometry Accurately Recognize Inclined Walking Terrains?
N. Wang, S. J. Redmond, E. Ambikairajah, B. G. Celler and N. H. Lovell, "Can Triaxial Accelerometry Accurately Recognize Inclined Walking Terrains?," in IEEE Transactions on Biomedical Engineering, vol. 57, no. 10, pp. 2506-2516, Oct. 2010.

In this work, a Gaussian Mixture Model was used to predict 7 different gradient surfaces. 12 subjects were tested in total and a set of 13 features were extracted for prediction. 

During walking experiments, a single triaxial accelerometer was placed at the waist, and video recordings were collected for segmenting the gait in post processing. The subjects were instructed to walk on paved ground or wair way. Each subject walked with their preferrered step length but at a constant step rate fixed by a metronome.

The 13 features were extracted from a variable length analysis window (approximately 3.46 seconds). Both global features, such as zero crossings and pairwise crossing relationships, and local features, such as heel strike timings, were used for prediction.

For the GMM, the optimal number of Gaussian functions required was determined to be 4. 100 iterations were used to maximize the expectation in training. The performance of the classifier was also evaluated using six-fold cross-validation. The data was divided into subsets without spliting data from each subject. 

Some of features used in this classification are reliant on accurate gait segmentation from motion anlysis, thus a separate gait segmentation algorithm was presented to validfy the feasibility of the slope classifier out of a laboratory.

### 3. Accelerometry Based Classification of Walking Patterns Using Time-frequency Analysis

N. Wang, E. Ambikairajah, N. H. Lovell and B. G. Celler, "Accelerometry Based Classification of Walking Patterns Using Time-frequency Analysis," 2007 29th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, Lyon, 2007, pp. 4899-4902.

The goal of this research is to classify 5 walking conditions, including stair ascent, stair descent, uphill, downhill and level ground. The method used is Multi-layer perception (MLP) neural networks (NNs) classifier. The result of the classifier showed quite high accurarcy: 88% for Round Robin train-test method and 92.05% for Random frame selecting train-test method.

### 4. Classification of Walking Patterns on Inclined Surfaces from Accelerometry Data

N. Wang, E. Ambikairajah, S. J. Redmond, B. G. Celler and N. H. Lovell, "Classification of walking patterns on inclined surfaces from accelerometry data," 2009 16th International Conference on Digital Signal Processing, Santorini-Hellas, 2009, pp. 1-4.

The goal of this research is to present a classification method to distinguish between 4 different gradient surfaces, 4.8% uphill, 4.8% downhill, 17.3% uphill and 17.3% downhill. Data of flat ground walking was also collected to provide a baseline example of eah subject's gait. The baseline was also used to normalize the data collected on gradient surfaces.

Using a GMM model on data from 12 participants, a classification accuracy of 90.9% was achieved.

### 5. A New Adaptive Frequency Oscillator for Gait Assistance

Keehong Seo, SeungYong Hyung, Byung Kwon Choi, Younbaek Lee and Youngbo Shim, "A new adaptive frequency oscillator for gait assistance," 2015 IEEE International Conference on Robotics and Automation (ICRA), Seattle, WA, 2015, pp. 5565-5571.

A adaptive frequency oscillator was developed to estimate gait cycle from foot contact sensors. This information is beneficial for controling exoskeletons for walking gait asistance in order to devliver synchronous assistance. 

Although walking slope estimation was not the main goal of this research, controller uses estimated slope to interpolate asistance torque profiles. The joint angle trajecteries, and cadence estimate by the adaptive frequency controller are used to estimate the walking slope and speed. The walking speed is predicted using a regression model using acceleration ata.



```python

```

