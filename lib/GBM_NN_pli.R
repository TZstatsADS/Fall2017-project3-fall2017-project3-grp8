setwd("Fall2017-project3-fall2017-project3-grp8/")

library(gbm)
library(dplyr)
library(neuralnet)

gbm_cv_param_optimization = function(train_data) {
  
  ntrees = c(100,300,500)
  interactiondepths = c(1,3,5)
  shrinkages = c(0.01,0.05,0.1)
  nfolds = 5
  
  niter = length(ntrees)*length(interactiondepths)*length(shrinkages)*nfolds
  cv_res = data.frame(ntree = rep(NA,niter),
                      interactiondepth = rep(NA,niter),
                      shrinkage = rep(NA,niter),
                      err = rep(NA,niter))
  cv_indices = sample(rep(c(1:nfolds),2600/nfolds),2600,replace=F)
  counter = 1
  
  for (i in 1:length(ntrees)) {
    for (j in 1:length(interactiondepths)) {
      for (k in 1:length(shrinkages)) {
        for (l in 1:nfolds) {
          print(paste0("i=",i," j=",j," k=",k," l=",l," counter=",counter))
          cv_res$ntree[counter] = ntrees[i]
          cv_res$interactiondepth[counter] = interactiondepths[j]
          cv_res$shrinkage[counter] = shrinkages[k]
          
          test.cv = train_data[cv_indices == l,]
          train.cv = train_data[cv_indices != l,]
          
          model = gbm(y~.,data=train.cv,
                      distribution='multinomial',
                      n.trees=ntrees[i],
                      interaction.depth=interactiondepths[j],
                      shrinkage=shrinkages[k])
          
          predBST.train = predict(model,n.trees=ntrees[i], newdata=test.cv[,-1],type='response')
          
          p.predBST.train <- apply(predBST.train, 1, which.max)
          
          cv_res$err[counter] = 1-sum(test.cv[,1] == p.predBST.train-1) / length(test.cv[,1])
          
          counter = counter + 1
          
        }
      }
    }
  }
  
  cv_agg = cv_res %>% 
    dplyr::group_by(ntree,interactiondepth,shrinkage) %>%
    dplyr::summarise(err = mean(err,na.rm=T))
  
  optim = cv_agg[which.min(cv_agg$err),]
  
  return(optim)
  
}

label = read.csv("data/our_data/training_set/label_train.csv")

set.seed(1)

#### GBM
## Sift Features
sift.features = read.csv("data/our_data/training_set/features_GIST.csv",header=F)
all=data.frame(cbind(label[,2],sift.features))
colnames(all)[1] = 'y'
test.index=sample(1:3000,400,replace=F)
test.sift=all[test.index,]
test.sift.actual = test.sift[,1]
test.x.sift=test.sift[,-1]
train.sift=all[-test.index,]

# optimal params
optim_sift = gbm_cv_param_optimization(train.sift)

model = gbm(y~.,data=train.sift,
            distribution='multinomial',
            n.trees=optim_sift$ntree,
            interaction.depth=optim_sift$interactiondepth,
            #cv.folds=5,
            shrinkage=optim_sift$shrinkage)

# test error
predBST = predict(model,n.trees=optim_sift$ntree, newdata=test.sift,type='response')
p.predBST <- apply(predBST, 1, which.max)
1-sum(test.sift.actual == (p.predBST-1)) / length(test.sift.actual)






#### NEURAL NET
nn.train = cbind(train.sift[,1] == 0, train.sift[,1]==1,train.sift[,1]==2,train.sift[,-1])
colnames(nn.train)[1:3] = c('y1','y2','y3')
n = colnames(nn.train)[-c(1:3)]
f <- as.formula(paste("y1+y2+y3 ~", paste(n[!n %in% "label_train"], collapse = " + ")))
model = neuralnet(f,
                  data = nn.train,
                  act.fct = "logistic",
                  hidden = 5,
                  linear.output = FALSE,
                  lifesign = "minimal")

pr.nn <- compute(model, test.sift[,-1])
pr.nn_ <- pr.nn$net.result

p.predBST <- apply(pr.nn_, 1, which.max)
1-sum(test.sift.actual == (p.predBST-1)) / length(test.sift.actual)

