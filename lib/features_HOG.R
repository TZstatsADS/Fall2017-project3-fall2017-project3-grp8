

setwd("/Users/duanshiqi/Documents/Github/Fall2017-project3-fall2017-project3-grp8/lib")
img_dir <- "../data/our_data/training_set/images/"
#dir_names <- list.files(img_dir)
#n_files <- length(dir_names)

#img0 <- readImage(paste0(img_dir, dir_names[1]))
#h1 <- HOG(img0)

feature<-function(img_dir, export=TRUE){
  
  ### Construct process features for training/testing images
  ### HOG: calculate the Histogram of Oriented Gradient for an image
  
  ### Input: a directory that contains images ready for processing
  ### Output: an .RData file contains processed features for the images
  
  
  ### load libraries
  library("EBImage")
  library("OpenImageR")
  
  dir_names <- list.files(img_dir)
  n_files <- length(dir_names)
  
  ### calculate HOG of images
  dat <- matrix(NA, n_files, 54) 
  for(i in 1:n_files){
    img <- readImage(paste0(img_dir,  dir_names[i]))
    dat[i,] <- HOG(img)
  }
  
  ### output constructed features
  if(export){
    save(dat, file=paste0("../output/HOG.RData"))
  }
  return(dat)
}

dat_HOG<-feature(img_dir)
