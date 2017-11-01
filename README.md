# Project: Dogs, Fried Chicken or Blueberry Muffins?
![image](figs/chicken.jpg)
![image](figs/muffin.jpg)

### [Full Project Description](doc/project3_desc.md)

Term: Fall 2017

+ Team 8
+ Team members
	+ Shiqi Duan 
	+ Jordan Leung
	+ Jingkai Li
	+ Peter Li
	+ Stephanie Park
	
+ Project summary: In this project, we created a classification engine for images of poodles, fried chickens, and blueberry muffins. We set our baseline model using SIFT features and gradient boosting machine(GBM) classifier. Besides the SIFT features, we also tried GIST and HOG (Histogram of oriented Gradient) descriptors. In terms of classifiers, we considered SVM(Linear and RBF kernel), Random Forest, XGBoost, Logistic Regression, Neural Network, and GBM. After model evaluation and comparison, the final advanced model we selected is using HOG descriptor and a majority vote of XGBoost, Linear SVM, and RBF-kernel SVM classifiers. We increased the accuracy by % and only took % of running time as in baseline model. 

+ The root code of our project is available at [Main.Rmd](doc/main.Rmd)
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members approve our work presented in this GitHub repository including this contributions statement. 

Part of the project each member is responsible for:
   
     Shiqi Duan: Feature Extraction,
		  Logistic Regression,
                 RBF-kernel SVM and Random Forest on HOG features,
                 Main File Construction

     Jordan Leung: XgBoost, Main File Construction, Project Presentation

     Jingkai Li: Linear SVM, 
		  RBF－kernel SVM on GIST and SIFT

     Peter Li: GBM, Neural Networks, Main File Construction
     
     Stephanie Park: Random Forest on GIST and SIFT, Project Presentation

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
