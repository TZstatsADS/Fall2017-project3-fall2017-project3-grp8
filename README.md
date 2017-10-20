Tasks (week Oct 19 - Oct 25):

Shiqi Duan: extract features using GIST (already done), HOG, and LBP. Train Logistic Regression classifier

Jordan Leung: Xgboost classier

Jinghai Li: SVM classifier with linear or non-linear kernel

Peter Li: GBM classifier and neural network (try)

Stephanie Park: Random Forest classifier

Note: there are four features SIFT, GIST, HOG, and LBP. Each person has to train their model on the four features. That means you should have four models: SIFT+classifier, GIST+classifier, HOG+classifier, and LBP+classifier. 

During training, need to use cross-validation (remember to set seed when randomly splitting the dataset into K-folds). You need to tune the parameters in the classifiers to achieve the smallest CV-training error and relatively short training time. Once you have got the best parameters for your model. Then you train the classifier on the whole training dataset to get the training time T and training error E. Remember to record them.

The main.R in doc and some .R files in lib can be used as references when you train your classifiers.


Go ahead buddies!



# Project: Dogs, Fried Chicken or Blueberry Muffins?
![image](figs/chicken.jpg)
![image](figs/muffin.jpg)

### [Full Project Description](doc/project3_desc.md)

Term: Fall 2017

+ Team #
+ Team members
	+ team member 1
	+ team member 2
	+ team member 3
	+ team member 4
	+ team member 5

+ Project summary: In this project, we created a classification engine for images of dogs versus fried chicken versus blueberry muffins. 
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
