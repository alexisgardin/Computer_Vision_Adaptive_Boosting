
# Computer_Vision_Adaptive_Boosting
Script for training and testing on Fashion Mnist dataset for Machine learning purpose
Based on https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html tutorial

<img src="result.png" width="100%">

# Prepare test and training data
Generate training and testing image from fashion mnist byte file
```
python3 fashionGenerateImage.py
```
Generate info.dat for positive sample  and bg.txt for negative sample with the command **labels** which allow you to choose what you want to detect
### Labels
Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |
### Example for T-shirt :
```
python3 fashionGenerateImage.py --labels 0
```
### Create positive sample
```
opencv_createsamples -info info.dat -vec positive.vec -w 28 -h 28 -num 6000
```
# How to train ?
2000 positive image and 2000 negative image are the default value and it took me 2 hours with 10 stages
```
 opencv_traincascade -data training -vec positive.vec  -bg bg.txt -numPos 2000 -numNeg 2000 -w 28 -h 28 -numStages 10
```
This will generate a cascade.xml file in the training folder, so you can now test whether your training has been effective

# How to test ?
We have training data, that we have generate before, to test we will use a script which give some information like


 - Sensibilité
 - Sépcificité
 - Précision
 - F-mesure

And a file **result.png**

### Example
```
python3 testCascade.py --labels 0 --cascade /path/to/your/cascade
```
Default value :

 - Labels = 0
 - Cascade path = training/cascade.xml
