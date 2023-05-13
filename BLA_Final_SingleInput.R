#########################################################################################################
# Initial Setup of working directory and functions used to calculate scores
#########################################################################################################

#Activation function between layers
activation <- function(x){
  ######## Leaky ReLu #######
  #tmp = NULL
  #eps = 0.0001
  #for(i in 1:length(x)){
  #  if(x[i]<0){
  #    tmp[i] = eps*x[i]
  #  } else{
  #    tmp[i]=x[i]
  #  }
  #}
  #return(tmp)
  ######## Inverse Hyperbolic Tangent #######
  value = tanh(x)
  return(value)
}

#Scoring function for sampling the weights
score <- function(x,epoch){
  ######## Uniform Weighting #######
  #rank = 1
  ######## Inverse Weighting #######
  #rank = min(1/(x^2),1000000)
  ######## Exponential Weighting #######
  rank = exp(-1*x^2)
  #rank = 1/cosh(x^2)
  return(rank)
}

#Curve to estimate
estimated_curve <- function(x){
  #return(x^3-2*x^2+5*x-1) #+ rnorm(1,0,0.25))
  #return(sin(x^2)-0.03*x^5)
  return(-(x-2)^3*(x+1)^2*(x-4)/8)
}

#Set up parameters for the algorithm
main <- function(epochs){
  #Architechture of neural network:
  number_of_inputs <<- 1
  number_of_hiddennodes1 <<- 100
  number_of_outputs <<- 1
  
  w1 <<- matrix(rnorm((number_of_inputs+1)*number_of_hiddennodes1,0,1), ncol = (number_of_inputs+1))
  w2 <<- matrix(rnorm((number_of_hiddennodes1+1)*number_of_outputs,0,1), ncol = (number_of_hiddennodes1+1))
  
  upperbound <<- 4
  lowerbound <<- -1
  
  #Reseting the accuracy so that we can track it throughout the algorithm
  it <<- 1
  
  #Number of splits in the dataset
  splits <<- c(10,5,4,3,2,rep(1,epochs-5))
  #splits <<- c(rep(1,epochs))
  
  #Sample Size
  smp_size <<- 6000
  
  
  #Creating the training set
  xtrain <<- seq(lowerbound,upperbound,length=smp_size)
  ytrain <<- sapply(xtrain,estimated_curve)
  
  #Normalize training set
  xtrain <<- -1 + 2*(xtrain-min(xtrain))/(max(xtrain)-min(xtrain))
  
  traindata <<- cbind(xtrain,ytrain)
}


splitsLog <- function(epochs){
  #Number of splits in the dataset
  splits <<- c(10,5,4,3,2,rep(1,epochs-5))
  #splits <<- c(rep(1,epochs))
  
  #Create log to store all MSE values
  mse_log <<- matrix(NA, nrow = sum(splits), ncol=0)
}



#########################################################################################################
# Run algorithm
#########################################################################################################

#Let v be the number of iterations you want to test the algorithm on
for(v in 1:1){
  #Number of epochs:
  epochs = 50
  delta = 40
  
  main(epochs)
  splitsLog(epochs)
  par(mfrow=c(1,2))
  meansquarederror = NULL
  
  for(epoch in 1:epochs){
    
    #Randomly sample the sampling set
    traindata = traindata[sample(1:nrow(traindata)),]
    
    #Size of sampling set:
    m <- smp_size/(splits[epoch])
    #Let k be the number of mini batches for the sampling set, this is not yet generalized and will not work with random values
    for(k in 1:splits[epoch]){
      #These values move along the data sets so that we see new observations throughout the process
      a = 1 + m*(k-1)
      b = m*k
      data1 = traindata[a:b,]
      
      #Tracking the observations and all their hidden layer values during the feed forward process.
      #This just saves us from having to calculate any inverses
      samplevalues = NULL
      
      #Feed forward through the network
      for(i in 1:m){
        singlesample = NULL
        x1 = c(1,as.numeric(data1[i,1]))
        y1 = w1 %*% x1
        x2 = c(1,sapply(y1,activation))
        y2 = w2 %*% x2
        truey2 = as.numeric(data1[i,2])
        singlesample = c(x1,y1,y2,truey2)
        samplevalues = rbind(samplevalues,singlesample)
      }
      
      batchsize=smp_size/splits[epoch]
      
      #Set all the initial matrices
      A1=0
      A2=0
      B1=0
      B2=0
      #Update the weights based on the values in the updating set
      for(i in 1:m){
        updatex1 = c(1,as.numeric(data1[i,1]))
        updatey2 = as.numeric(data1[i,2])
        
        distances = abs(updatex1[2]-samplevalues[,2])
        samples = sort(distances, index.return=TRUE, decreasing=FALSE)
        nn = samples$ix[1:delta]
        
        tmpvalues = samplevalues[nn,]
        
        errors = NULL
        for(y in 1:length(nn)){
          errors[y] = ((updatex1[2]-tmpvalues[y,2])^2+(updatey2-tmpvalues[y,(ncol(tmpvalues)-1)])^2)/2
        }
        tmpvalues = cbind(tmpvalues, errors)
        
        #Set the lowest error to zero, apply the scoring function and sample
        regularized = tmpvalues[,ncol(tmpvalues)]-min(tmpvalues[,ncol(tmpvalues)])
        probs = sapply(regularized,score,epoch=epoch)
        probs = probs/sum(probs)
        point = sample(1:length(nn),1,prob = probs)
        
        
        #Find the internal values of the nodes
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
          print("test")
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
      
      gain1 = 1.95/max(eigen(A1check)$values)
      gain2 = 1.95/max(eigen(A2check)$values)
      
      for(s in 1:100000){
        w1 = w1 + gain1*(B1check-w1%*%A1check)
        w2 = w2 + gain2*(B2check-w2%*%A2check)
      }
    }
    
    #Graph results
    if(epoch %in% c(1:50)){
      tmp = seq(lowerbound,upperbound,length=10000)
      tmpnorm = -1 + 2*(tmp-min(tmp))/(max(tmp)-min(tmp))
      estimates = NULL
      sqerror = 0
      for (s in 1:length(tmpnorm)){
        x1 = c(1,tmpnorm[s])
        X1tmp = w1 %*% x1
        x2 = c(1,sapply(X1tmp,activation))
        output = w2 %*% x2
        estimates = c(estimates,output)
        sqerror = sqerror + (output-(sapply(tmp[s],estimated_curve)))^2
      }
      meansquarederror = c(meansquarederror, sqerror/length(tmp))
      plot(tmp,sapply(tmp,estimated_curve), type = "l",xlim = c(lowerbound,upperbound), main = "Estimated Curved vs True Curve", xlab = "Inputs", ylab = "Predicted Values")
      lines(tmp,estimates,type="l", col="blue")
      legend("bottom", legend = c("True Values", "NN Values"), col = c("black", "blue"), lty=1)
      plot(meansquarederror, type = "l", main = "Mean Squared Error for Dataset", xlab = "Epoch", ylab = "MSE")
    }
  }

  cat("Iteration: ", v, " complete")
  mse_log = cbind(mse_log,meansquarederror)
  print(rowMeans(mse_log))
}
print(rowMeans(mse_log))
