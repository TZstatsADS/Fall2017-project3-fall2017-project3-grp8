# load library
library(e1071)

### HOG features ###

# load data
setwd("/Users/duanshiqi/Documents/Github/Fall2017-project3-fall2017-project3-grp8/lib")
y<-read.csv("../data/our_data/training_set/label_train.csv",header=TRUE,as.is=TRUE)[,-1]
X<-read.csv("../data/our_data/training_set/features_HOG.csv",header=TRUE,as.is=TRUE)[,-1]

y<-as.factor(y)

# tune parameters and tune control
par.list = list(cost = c(0.001, 0.01, 0.1, 10, 100, 1000, 5000, 10000),
                 gamma = c(0.001, 0.005, 0.01, 0.05, 1, 5, 8, 10))
k = tune.control(cross = 5)

# tune svm with multiple classes using the one-versus-one approach
set.seed(0)
tune.out = tune(svm, train.x = X, train.y = y, kernel = "radial",
                scale = FALSE, ranges = par.list, tunecontrol = k)

tune.out$best.parameters # cost = 10000, gamma =10
tune.out$best.performance # 0.197
save(tune.out, file="../output/fit_train_svmRBF_HOG.RData")

# train the best model on the whole training set
bestmod = tune.out$best.model
tm_train <- system.time(pred <- predict(bestmod, X))
save(pred, file="../output/pred_test_svmRBF_HOG.RData")

sum(pred != y)/3000
cat("Time for training model=", tm_train[1], "s \n") # 0.497s


