library(matrixStats)
library(datasets)
library(caret)
data("iris")

iris$setosa <- as.numeric(iris$Species == "setosa")
iris$versicolor <- as.numeric(iris$Species == "versicolor")
iris$virginica <- as.numeric(iris$Species == "virginica")
iris <- iris[, -5]

softmax <- function(x) {
  exp_x <- exp(x-max(x))  # Subtract max(x) for numerical stability
  probabilities <- exp_x / sum(exp_x)
  return(probabilities)
}


#Activation function between layers
activation <- function(x){
  # tmp = NULL
  # eps = 0.0001
  # if(x<0){
  #   tmp = eps*x
  # } else{
  #   tmp=x
  # }
  # return(x)
  value = tanh(x)
  return(value)
}

#Function to create probability distribution
score <- function(x,epoch){
  #Uniform
  #rank = 1
  #Inverse
  #rank = min(1/x*epoch,1000000)
  #exponential
  rank = exp(-x/3)
  #rank= -x
  #rank = 1/cosh(x^2)
  return(rank)
}

#Note that when using a single variable case nrow(y) needs to be changed to length(y)
nearestneighbours <- function(x,y,n){
  distances = NULL
  for(i in 1:nrow(y)){
    distances[i] = sum(abs(x-y[i,]))
  }
  samples = NULL
  for (i in 1:n){
    samples[i] = which.min(distances)
    distances[samples[i]] = 10000000000000
  }
  return(samples)
  #lst = sort(distances, index.return = TRUE, decreasing = FALSE)
  #return(lst$ix[1:n])
}


main <- function(epochs){
  #Architechture of neural network:
  number_of_inputs <<- 4
  number_of_hiddennodes1 <<- 100
  number_of_outputs <<- 3
  
  #Random start points for the weights
  w1 <<- matrix(rnorm((number_of_inputs+1)*number_of_hiddennodes1,0,0.1), ncol = (number_of_inputs+1))
  w2 <<- matrix(rnorm((number_of_hiddennodes1+1)*number_of_outputs,0,0.1), ncol = (number_of_hiddennodes1+1))
  
  w1check <<- w1
  w2check <<- w2
  
  
  
  # Split the dataset into 80% train and 20% test
  train_indices <- sample(1:nrow(iris),120)
  
  # Create train and test datasets
  traindata <<- iris[train_indices, ]
  testdata <<- iris[-train_indices, ]
  
  traindata$setosa <<- ifelse(traindata$setosa == 1, log(0.9999999), log(0.0000001)-log(2))
  traindata$versicolor <<- ifelse(traindata$versicolor == 1, log(0.9999999), log(0.0000001)-log(2))
  traindata$virginica <<- ifelse(traindata$virginica == 1, log(0.9999999), log(0.0000001)-log(2))
  
  
  #Normalizing the inputs for the x variable training set
  for(i in 1:4){
    traindata[,i] <<- -1 + 2*(traindata[,i]-min(traindata[,i]))/(max(traindata[,i])-min(traindata[,i]))
  }
  
  
  #Normalizing the inputs for the x variable training set
  for(i in 1:4){
    testdata[,i] <<- -1 + 2*(testdata[,i]-min(testdata[,i]))/(max(testdata[,i])-min(testdata[,i]))
  }
  
  testdata$y1 <<- ifelse(testdata$setosa == 1, log(0.9999999), log(0.0000001)-log(2))
  testdata$y2 <<- ifelse(testdata$versicolor == 1, log(0.9999999), log(0.0000001)-log(2))
  testdata$y3 <<- ifelse(testdata$virginica == 1, log(0.9999999), log(0.0000001)-log(2))
  
}

splitsLog <- function(epochs){
  #Number of splits in the dataset
  #splits <<- c(10,5,4,3,2,rep(1,epochs-5))
  splits <<- c(rep(1,epochs))
  
}

#########################################################################################################
# Run algorithm
#########################################################################################################

#Number of epochs:
epochs = 5
#Create log to store all MSE values
accuracy_log <<- matrix(NA, nrow = epochs, ncol=0)
mse_log <<- matrix(NA, nrow = epochs, ncol=0)

for(v in 1:200){
  par(mfrow=c(1,2))
  delta = 10
  
  main(epochs)
  splitsLog(epochs)
  meansquarederror = NULL
  accuracy = NULL
  for(epoch in 1:epochs){
    
    #Randomly sample the sampling set
    traindata = traindata[sample(1:nrow(traindata)),]
    
    #Size of sampling set:
    batchsize=floor(nrow(traindata)/splits[epoch])
    
    for(k in 1:splits[epoch]){
      #These values move along the data sets so that we see new observations throughout the process
      a = 1 + batchsize*(k-1)
      b = batchsize*k
      data1 = traindata[a:b,]
      
      #Tracking the observations and all their hidden layer values during the feed forward process.
      #This just saves us from having to calculate any inverses
      samplevalues = NULL
      
      for(i in 1:batchsize){
        singlesample = NULL
        x1 = c(1,as.numeric(data1[i,1:number_of_inputs]))
        y1 = w1check %*% x1
        x2 = c(1,sapply(y1,activation))
        y2 = w2check %*% x2
        truey2 = as.numeric(data1[i,(number_of_inputs+1):(number_of_inputs+3)])
        singlesample = c(x1,y1,y2,truey2)
        samplevalues = rbind(samplevalues,singlesample)
      }
      
      #Set all the initial matrices
      A1=0
      A2=0
      B1=0
      B2=0
      #Updating the samples one at a time at this point using the sampling set to bootstrap the hiddenlayer
      for(i in 1:batchsize){
        #Reading in the data for the update sample
        updatex1 = c(1,as.numeric(data1[i,1:number_of_inputs]))
        updatey2 = as.numeric(data1[i,-c(1:number_of_inputs)])
        #Finding the N nearest neighbours with respect to the L1 norm for the inputs
        # distances = rowSums(abs(updatex1[2:(number_of_inputs+1)]-samplevalues[,2:(number_of_inputs+1)]))
        # samples = sort(distances, index.return=TRUE, decreasing=FALSE)
        # value = max(delta+1-epoch,8)
        # nn = samples$ix[1:value]
        
        correctclass = NULL
        
        for(i in 1:batchsize){
          
          if (identical(samplevalues[i,(ncol(samplevalues)-number_of_outputs+1):ncol(samplevalues)], updatey2)) {
            correctclass <- rbind(correctclass,samplevalues[i,])
          }
        }
        
        
        nn = nearestneighbours(updatex1[2:(number_of_inputs+1)], correctclass[,2:(number_of_inputs+1)],max(delta-epoch,5))
        
        tmpvalues = correctclass[nn,]
        
        #Finding the errors of that point with respect to the L1 norm for the outputs
        errors = NULL
        for(y in 1:length(nn)){
          max_index <- which.max(updatey2)
          result_vector <- c(0,0,0)
          result_vector[max_index] <- 1
          errors[y] = -1*sum(updatey2*softmax(tmpvalues[y,(ncol(tmpvalues)-5):(ncol(tmpvalues)-3)]))
        }
        tmpvalues = cbind(tmpvalues, errors)
        
        #Subtracting off the min error so we don't get floating point errors at a later step
        regularized = tmpvalues[,ncol(tmpvalues)]-min(tmpvalues[,ncol(tmpvalues)])
        #Applying these errors to the scoring function and creating a probability distribution out of them
        probs = sapply(regularized,score,epoch=epoch)
        probs = probs/sum(probs)
        #Sampling a point from the N nearest neighbours using that probability distribution
        point = sample(1:length(nn),1,prob = probs)
        
        #Plugging the hidden layer values of that sample point into the middle of the update sample
        updatey1 = as.numeric(tmpvalues[point,(2+number_of_inputs):(number_of_inputs+number_of_hiddennodes1+1)]) 
        updatex2 = c(1,sapply(updatey1,activation))
        
        A1 = A1 + updatex1 %*% t(updatex1)
        A2 = A2 + updatex2 %*% t(updatex2)
        B1 = B1 + updatey1 %*% t(updatex1)
        B2 = B2 + updatey2 %*% t(updatex2)
      }
      if(epoch == 1){
        if(k==1){
          A1check = A1/batchsize
          A2check = A2/batchsize
          B1check = B1/batchsize
          B2check = B2/batchsize
        } else{
          A1check = A1check/2 + A1/(2*batchsize)
          A2check = A2check/2 + A2/(2*batchsize)
          B1check = B1check/2 + B1/(2*batchsize)
          B2check = B2check/2 + B2/(2*batchsize)
        }
      } else{
        A1check = A1check/2 + A1/(2*batchsize)
        A2check = A2check/2 + A2/(2*batchsize)
        B1check = B1check/2 + B1/(2*batchsize)
        B2check = B2check/2 + B2/(2*batchsize)
      }
      
      w1=w1check
      w2=w2check
      
      
      gain1 = 2/(max(eigen(A1check)$values)+min(eigen(A1check)$values))
      gain2 = 2/(max(eigen(A2check)$values)+min(eigen(A2check)$values))
      
      for(s in 1:100000){
        w1 = w1 + gain1*(B1check-w1%*%A1check)
        w2 = w2 + gain2*(B2check-w2%*%A2check)
      }
      
      #w1check = w1check/2 +w1/(2)
      #w2check = w2check/2 +w2/(2)
      w1check = w1
      w2check = w2
    }
    
    
    
    #Graph results
    correct <- 0
    mse <- 0
    for(s in 1:nrow(testdata)){
      x1 = as.numeric(c(1,testdata[s,1:4]))
      X1tmp = w1check %*% x1
      x2 = c(1,sapply(X1tmp,activation))
      y2 = w2check %*% x2
      output= softmax(y2)
      max_index <- which.max(output)
      result_vector <- c(0,0,0)
      result_vector[max_index] <- 1
      if (identical(result_vector, as.numeric(testdata[s,5:7]))) {
        correct <- correct + 1
      }
      mse = mse + sum((y2-testdata[s,8:10])^2)
    }
    accuracy = c(accuracy, correct/nrow(testdata)*100)
    meansquarederror = c(meansquarederror, mse/nrow(testdata))
    cat(correct, "out of", nrow(testdata), ":", correct/nrow(testdata)*100, "\n\n")
    plot(accuracy, type = "l", main = "Accuracy for Dataset")
    plot(meansquarederror, type = "l", main = "Mean Squared Error for Dataset")
  }
  
  cat("Iteration: ", v, " complete")
  accuracy_log = cbind(accuracy_log,accuracy)
  mse_log = cbind(mse_log,meansquarederror)
  print(rowMeans(accuracy_log))
  print(rowMeans(mse_log))
}
print(rowMeans(accuracy_log))
print(rowMeans(mse_log))

write.csv(accuracy_log, "C:\\Users\\sjsty\\Desktop\\Masters\\Final Programs\\Iris_3step_CE2.csv", row.names=FALSE)
