library(neuralnet)

train_nn = function(train_data) {
  
  nn.train = cbind(train_data[,1] == 0, train_data[,1]==1,train_data[,1]==2,train_data[,-1])
  colnames(nn.train)[1:3] = c('y1','y2','y3')
  n = colnames(nn.train)[-c(1:3)]
  
  f <- as.formula(paste("y1+y2+y3 ~", paste(n[!n %in% "label_train"], collapse = " + ")))
  model = neuralnet(f,
                    data = nn.train,
                    act.fct = "logistic",
                    hidden = 5,
                    linear.output = FALSE,
                    lifesign = "minimal")
  
  return(model)
  
}