########################
### Cross Validation ###
########################

### Author: Yuting Ma
### Project 3
### ADS Spring 2016


cv.gbm <- function(data, n_folds, par){
  
  indices = sample(x = rep(c(1:n_folds),nrow(data)/n_folds),
                   size = nrow(data),
                   replace = F)
  
  err = rep(NA,n_folds)  

  for (i in 1:n_folds) {
    
    model = train.gbm(train.data=data[which(indices != i),],
                               par = par)
    
    test_data = data[which(indices == i),]
    
    res = test.gbm(fit_train = model, test.data = test_data[,-1])
    
    err[i] = 1 - sum(res == test_data[,1]) / nrow(test_data)
    
  }

  return(c(mean(err),sd(err)))
  
}
