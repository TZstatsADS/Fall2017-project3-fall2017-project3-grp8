########################
### Cross Validation ###
########################

cv.gbm <- function(dat_train, par, K){
  
  n <- nrow(dat_train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold))) 
  
  err.gbm = rep(NA, K)  

  for (i in 1:K) {
    fit_gbm = train.gbm(train.data = dat_train[s != i,], par = par)
    
    pred = test.gbm(fit_train = fit_gbm, data.test = dat_train[s == i, -1])
    err.gbm[i] = mean(pred != dat_train[s == i, 1])
  }

  return(c(mean(err.gbm),sd(err.gbm)))
}


cv.svm_rbf <- function(dat_train, par.list, K){
  # tune svm with multiple classes using the one-versus-one approach
  tune.out = tune(svm, train.x = dat_train[, -1], train.y = dat_train[, 1], 
                  kernel = "radial",
                  scale = FALSE, 
                  ranges = par.list, 
                  tunecontrol = tune.control(cross = K))
  
  best.para = tune.out$best.parameters 
  smallest.cv_error = tune.out$best.performance
  performance.tune <- tune.out$performances
  
  return(list(best.par = best.para, 
              smallest.cv_error = smallest.err, 
              performances = performance.tune))
}


cv.svm_linear <- function(dat_train, par.list, K){
  # tune svm with multiple classes using the one-versus-one approach
  tune.out = tune(svm, train.x = dat_train[, -1], train.y = dat_train[, 1], 
                  kernel = "linear",
                  scale = FALSE, 
                  ranges = par.list, 
                  tunecontrol = tune.control(cross = K))
  
  best.para = tune.out$best.parameters 
  smallest.cv_error = tune.out$best.performance
  performance.tune <- tune.out$performances
  
  return(list(best.par = best.para, 
              smallest.cv_error = smallest.err, 
              performances = performance.tune))
}
