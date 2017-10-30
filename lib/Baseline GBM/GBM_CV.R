library(gbm)
library(dplyr)

source("lib/Baseline GBM/GBM_Train.R")
source("lib/Baseline GBM/GBM_Test.R")

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
          
          
          model = train_baseline_gbm(data = train.cv,
                                     n_trees = ntrees[i],
                                     interaction_depth = interactiondepths[j],
                                     shrinkage = shrinkages[k])
          
          p.predBST.train = test_baseline_gbm(data = test.cv[,-1],
                                              model = model,
                                              n_trees = ntrees[i])
          
          cv_res$err[counter] = 1-sum(test.cv[,1] == p.predBST.train) / length(test.cv[,1])
          
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