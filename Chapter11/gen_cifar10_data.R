library(keras)
library(imager)
# this script loads the cifar_10 data from Keras
#  and saves the data as individual images

# create directories,
#  we will save 8 classes in the data1 folder for model building
#  and use 2 classes for transfer learning
data_dir <- "../data/cifar_10_images/"
if (!dir.exists(data_dir))
  dir.create(data_dir)
if (!dir.exists(paste(data_dir,"data1/",sep="")))
  dir.create(paste(data_dir,"data1/",sep=""))
if (!dir.exists(paste(data_dir,"data2/",sep="")))
  dir.create(paste(data_dir,"data2/",sep=""))
train_dir1 <- paste(data_dir,"data1/train/",sep="")
valid_dir1 <- paste(data_dir,"data1/valid/",sep="")
train_dir2 <- paste(data_dir,"data2/train/",sep="")
valid_dir2 <- paste(data_dir,"data2/valid/",sep="")

if (!dir.exists(train_dir1))
  dir.create(train_dir1)
if (!dir.exists(valid_dir1))
  dir.create(valid_dir1)
if (!dir.exists(train_dir2))
  dir.create(train_dir2)
if (!dir.exists(valid_dir2))
  dir.create(valid_dir2)

# load CIFAR10 dataset
c(c(x_train,y_train),c(x_test,y_test)) %<-% dataset_cifar10()
# get the unique categories,
# note that unique does not mean ordered!
# save 8 classes in data1 folder
categories <- unique(y_train)
for (i in categories[1:8])
{
  label_dir <- paste(train_dir1,i,sep="")
  if (!dir.exists(label_dir))
    dir.create(label_dir)
  label_dir <- paste(valid_dir1,i,sep="")
  if (!dir.exists(label_dir))
    dir.create(label_dir)
}
# save 2 classes in data2 folder
for (i in categories[9:10])
{
  label_dir <- paste(train_dir2,i,sep="")
  if (!dir.exists(label_dir))
    dir.create(label_dir)
  label_dir <- paste(valid_dir2,i,sep="")
  if (!dir.exists(label_dir))
    dir.create(label_dir)
}

# loop through train images and save in the correct folder
for (i in 1:dim(x_train)[1])
{
  img <- x_train[i,,,]
  label <- y_train[i,1]
  if (label %in% categories[1:8])
    image_array_save(img,paste(train_dir1,label,"/",i,".png",sep=""))
  else
    image_array_save(img,paste(train_dir2,label,"/",i,".png",sep=""))
  if ((i %% 500)==0)
    print(i)
}

# loop through test images and save in the correct folder
for (i in 1:dim(x_test)[1])
{
  img <- x_test[i,,,]
  label <- y_test[i,1]
  if (label %in% categories[1:8])
    image_array_save(img,paste(valid_dir1,label,"/",i,".png",sep=""))
  else
    image_array_save(img,paste(valid_dir2,label,"/",i,".png",sep=""))
  if ((i %% 500)==0)
    print(i)
}

# plot some images to verify process
image_dir <- list.dirs(valid_dir1, full.names=FALSE, recursive=FALSE)[1]
image_dir <- paste(valid_dir1,image_dir,sep="")
img_paths <- paste(image_dir,list.files(image_dir),sep="/")

par(mfrow = c(3, 3))
par(mar=c(2,2,2,2))
for (i in 1:9)
{
  im <- load.image(img_paths[i])
  plot(im)
}
