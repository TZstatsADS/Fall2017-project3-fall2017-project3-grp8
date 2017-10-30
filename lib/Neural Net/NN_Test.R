library(neuralnet)

test_nn = function(model,
                   test_data) {
  
  pr.nn <- compute(model, test_data)
  pr.nn_ <- pr.nn$net.result
  
  p.predBST <- apply(pr.nn_, 1, which.max)
  
  return(p.predBST)
  
}