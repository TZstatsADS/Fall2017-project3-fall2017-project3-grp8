########################
### Cross Validation ###
########################
cv.sample <- function(dat_train, K){
  
  n <- nrow(dat_train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  return(s)
}

cv.gbm <- function(dat_train, par, K, idx){
  
  err.gbm = rep(NA, K)  

  for (i in 1:K) {
    
    fit_gbm = train.gbm(train.data = dat_train[idx != i,],
                               par = par)
    
    #test_data = dat[which(indices == i),]
    
    pred = test.gbm(fit_train = fit_gbm, data.test = dat_train[idx == i, -1])
    
    err.gbm[i] = mean(pred != dat_train[idx == i, 1])
  }

  return(c(mean(err.gbm),sd(err.gbm)))
  
}
