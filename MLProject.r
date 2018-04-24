library(randomForest) #Classification and regression model
library(miscTools) #Extracting the standard errors, obtaining the number of (estimated) parameters.
library(caret) #Training and plotting classification and regression models
library(ROCR) #For graphs, sensitivity/specificity curves, lift charts, and precision/recall plots
library(pROC) #visualizing, smoothing and comparing receiver operating characteristic (ROC curves)
library(e1071) #For latent class analysis
library(C50) #Decision trees and rule-based models for pattern recognition
library(rpart) #Recursive partitioning for classification, regression and survival trees
library(rpart.plot) #Plot rpart models
library(rattle) #Load data from a CSV file
library(RColorBrewer) #Provides color schemes for maps
library(ggplot2) #Creating graphics

#Loading the dataset.
news <- read.csv(file="C:/Users/user/Desktop/MLFinalProject/OnlineNewsPopularity.csv", head=TRUE, sep= ",")
dataset <- news
summary(news)

#Deleting URL and timedelta columns
newsreg <- subset( news, select = -c(url, timedelta ) )

#Standardize the data
#Generate z-scores using the scale() function
for(i in ncol(newsreg)-1){ 
  newsreg[,i]<-scale(newsreg[,i], center = TRUE, scale = TRUE)
}

#Define articles with shares larger than 1400 (median) as popular article
# Dataset for classification
newscla <-newsreg
newscla$shares <- as.factor(ifelse(newscla$shares > 1400,1,0))

#Split the data train 70% test 30%
#set random situation
set.seed(100)
#Select traning data and prediction data
ind<-sample(2,nrow(newscla),replace=TRUE,prob=c(0.7,0.3))

#Color palatte
color.knn<-'#00B2FF'#blue for KNN
color.cart<-'#FF1300' #red for cart
color.c50<-'#FFDC00' #yellow for c50
color.rf<-'#00FF49' #green for random forest

####KNN####
newscla.knn <- knn3(shares ~.,newscla[ind==1,])
newscla.knn.pred <- predict( newscla.knn,newscla[ind==2,],type="class")
newscla.knn.prob <- predict( newscla.knn,newscla[ind==2,],type="prob")

# Confusion matrix
confusionMatrix(newscla.knn.pred, newscla[ind==2,]$shares)

# ROC Curve
newscla.knn.roc <- roc(newscla[ind==2,]$shares,newscla.knn.prob[,2])
plot(newscla.knn.roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col=color.knn, print.thres=TRUE)

####CART###
newscla.cart<-rpart(shares ~.,newscla[ind==1,],method='class')
fancyRpartPlot(newscla.cart) #Generating nodes
summary(newscla.cart)

# ROC Curve
newscla.cart.roc <- roc(newscla[ind==2,]$shares,newscla.cart.prob[,2])
plot(newscla.cart.roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col=color.cart, print.thres=TRUE)

####C5.0####
newscla.c50<-C5.0(shares ~.,newscla[ind==1,],trials=5)
summary(newscla.c50)

#predict
newscla.c50.pred<-predict( newscla.c50,newscla[ind==2,],type="class" )
newscla.c50.prob<-predict( newscla.c50,newscla[ind==2,],type="prob" )
# Confusion matrix
confusionMatrix(newscla.c50.pred, newscla[ind==2,]$shares)

# ROC Curve
newscla.c50.roc <- roc(newscla[ind==2,]$shares,newscla.c50.prob[,2])
plot(newscla.c50.roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col=color.c50, print.thres=TRUE)

# Precision/Recall graph
newscla.c50.pred.pred <- prediction(newscla.c50.prob[,2],newscla[ind==2,]$shares)
newscla.c50.pred.perf <- performance(newscla.c50.pred.pred,"prec","rec")
plot(newscla.c50.pred.perf, avg= "threshold", colorize=T, lwd= 3,
     main= "... Precision/Recall graphs ...")
plot(newscla.c50.pred.perf, lty=3, col="grey78", add=T)

####Random Forest####
newscla.rf<-randomForest(shares ~.,newscla[ind==1,],ntree=100,nPerm=10,mtry=3,proximity=TRUE,importance=TRUE)
summary(newscla.rf)

# Plotting the number of trees vs error
plot(newscla.rf)

# Feature importance
newscla.rf.imp <- importance(newscla.rf)
newscla.rf.impvar <- newscla.rf.imp[order(newscla.rf.imp[, 3], decreasing=TRUE),]
newscla.rf.impvar

# Plot feature importance
varImpPlot(newscla.rf)

# Partial dependence
partialPlot(newscla.rf, newscla[ind==1,], kw_avg_avg, "0", main='' , xlab='kw_avg_avg', ylab="Variable effect")

#predict
newscla.rf.pred<-predict( newscla.rf,newscla[ind==2,], type="class")
newscla.rf.prob<-predict( newscla.rf,newscla[ind==2,], type="prob")
# Confusion matrix
confusionMatrix(newscla.rf.pred, newscla[ind==2,]$shares)

# ROC Curve
newscla.rf.roc <- roc(newscla[ind==2,]$shares,newscla.rf.prob[,2])
plot(newscla.rf.roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col=color.rf, print.thres=TRUE)

#Comparing all the models#
ROCCurve<-par(pty = "s")
plot(performance(prediction(newscla.knn.prob[,2],newscla[ind==2,]$shares),'tpr','fpr'),
     col=color.knn, lwd=3
)
text(0.55,0.6,"KNN",col=color.knn)
plot(performance(prediction(newscla.cart.prob[,2],newscla[ind==2,]$shares),'tpr','fpr'),
     col=color.cart, lwd=3, add=TRUE
)
text(0.3,0.4,"CART",col=color.cart)
plot(performance(prediction(newscla.c50.prob[,2],newscla[ind==2,]$shares),'tpr','fpr'),
     col=color.c50, lwd=3, add=TRUE
)
text(0.15,0.5,"C5.0",col=color.c50)
plot(performance(prediction(newscla.rf.prob[,2],newscla[ind==2,]$shares),'tpr','fpr'),
     col=color.rf, lwd=3, add=TRUE
)
text(0.3,0.7,"Random Forest",col=color.rf)