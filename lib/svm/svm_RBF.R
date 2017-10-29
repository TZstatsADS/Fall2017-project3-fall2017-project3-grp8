# load library
library(e1071)

### HOG features ###

# load data
setwd("/Users/duanshiqi/Documents/Github/Fall2017-project3-fall2017-project3-grp8/lib")
y<-read.csv("../data/our_data/training_set/label_train.csv",header=TRUE,as.is=TRUE)[,-1]
X<-read.csv("../data/our_data/training_set/features_HOG.csv",header=TRUE,as.is=TRUE)[,-1]

y<-as.factor(y)

# split into training set and validation(test) set
proportion = 0.75 # training set proportion
set.seed(0)
n = length(y)
index <- sample(n, n*proportion, replace = FALSE)

x.train <- X[index,]
y.train <- y[index]

x.test <- X[-index,]
y.test <- y[-index]

# tune parameters and tune control
par.list = list(cost = c(0.001, 0.01, 0.1, 1, 2, 3, 5, 8, 10),
                 gamma = c(10, 100, 200, 400, 450, 1000, 1200))
k = tune.control(cross = 5)

# tune svm with multiple classes using the one-versus-one approach
set.seed(0)
tune.out = tune(svm, train.x = x.train, train.y = y.train, kernel = "radial",
                scale = FALSE, ranges = par.list, tunecontrol = k)

best.para = tune.out$best.parameters # cost = 8, gamma =450 in fact, cost=10, gamma=100 already 0.2153
best.para
tune.out$best.performance # 0.19467
#performance.tune<-tune.out$performances

# train the best model on the training set
#bestmod = tune.out$best.model
tm_train <- system.time(fit_train<-svm(x.train,y.train,
                                       scale=FALSE,
                                       kernel="radial",
                                       gamma = best.para[2],
                                       cost = best.para[1]))
pred.train <- predict(fit_train, x.train)
train.error <- mean(pred.train != y.train)
save(fit_train, file="../output/fit_train_svmRBF_HOG.RData")

# test the best model on the test set
tm_test <- system.time(pred.test <- predict(fit_train, x.test))
test.error <- mean(pred.test != y.test)
save(pred.test, file="../output/pred_test_svmRBF_HOG.RData")

### Summarize Running Time

cat("Time for training model=", tm_train[1], "s \n") # 1.797s
cat("Time for making prediction=", tm_test[1], "s \n")


