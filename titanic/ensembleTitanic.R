
library('caret')
library('mlbench')
library('pROC')

set.seed(107)
inTrain <- createDataPartition(y = trainSet$Survived, p = .75, list = FALSE)
training <- trainSet[ inTrain,]
testing <- trainSet[-inTrain,]
my_control <- trainControl(
  method='boot',
  number=25,
  savePredictions=TRUE,
  classProbs=TRUE,
  index=createResample(training$Survived, 25),
  summaryFunction=twoClassSummary
)
#summaryFunction:  compute measures specific to two–class problems, such as the area under the ROC curve, the sensitivity and specificity
#the default is defaultSummary. When twoClassSummary is used, classProbs must be set to T

# Notice that we are explicitly setting the resampling index to being used in trainControl. 
#If you do not set this index manually, caretList will attempt to set it for automatically, 
# but it’s generally a good idea to set it yourself.

#use caretList to fit a series of models (each with the same trControl):
library(Hmisc)
library(rpart)
library(rattle)
library(rpart.plot)
library(randomForest)
library(party)
library(vcd)
library(ggplot2)
library(ggthemes)
library(caret)
library(e1071)
library(pROC)
library(ada)
library(kernlab)
library(gbm)
library(glmnet)
library(caretEnsemble)
library(RColorBrewer)
library(xtable)
library(pbapply)
library(nnls)

library(doMC) #doMC package to take advantage of parallel processing with multiple cores
registerDoMC(cores=4)

model_list <- caretList(
  Survived ~., data=training,
  trControl=my_control,
  metric = "Accuracy", 
  methodList=c( 'gbm','svmRadial','LogitBoost','glmnet','rf')
)

# Get more control w/tuneList()
# This argumenent can be used to fit several different variants of the same model, 
# and can also be used to pass arguments through train down to the component functions

pp = c("center", "scale") #simple preprocessing

# Set up models
# ----------------------------------------------------------------------------
# alpha controls relative weight of lasso and ridge constraints (1=lasso, 0=ridge)
# lambda is the regularization parameter
glmnetgrid = expand.grid(.alpha = seq(0, 1, 0.1), .lambda = seq(0, 1, 0.1))
#rfgrid = data.frame(.mtry = c(2,3)) 
rfgrid = expand.grid(.mtry = c(2,3)) #should be about sqrt(numPredictors)
#gbmgrid = expand.grid(.interaction.depth = c(1, 5, 9), .n.trees = (1:15)*100, .shrinkage = 0.1,n.minobsinnode =10) #og
gbmgrid = expand.grid(interaction.depth = c(1, 5, 9),n.trees = (1:30)*50,shrinkage = 0.1, n.minobsinnode = 10)
adagrid = expand.grid(.iter = c(50, 100), .maxdepth = c(4, 8), .nu = c(0.1, 1))
svmgrid = expand.grid(.sigma=c(0.1, 0.25, 0.5, 0.75, 1), .C=c(0.1, 1, 2, 5, 10))
cforgrid = expand.grid(mtry = c(2,3)) # not using
cforcontrols = cforest_unbiased(ntree = 2000)
blackgrid = expand.grid(.mstop=c(500), .maxdepth=c(5,10))
earthgrid = expand.grid(.nprune=c(10), .degree=c(1,2))
gambogrid = expand.grid(.mstop=c(500), .prune=c(10))
logitgrid = expand.grid(.nIter=c(10,50,100))

model_list_big <- caretList(
  Survived ~., data=training,
  trControl=my_control,
  metric='ROC',
  methodList=c('glmnet', 'rf', 'gbm','ada','svmRadial','cforest','blackboost','earth','gamboost','LogitBoost','bayesglm'),
  tuneList=list(
    glmnet=caretModelSpec(method='glmnet', tuneGrid=glmnetgrid,preProcess=pp),
    rf=caretModelSpec(method='rf', tuneGrid=rfgrid, preProcess='pca', ntree =2000 ),
    gbm=caretModelSpec(method='gbm', tuneGrid=gbmgrid),
    #ada=caretModelSpec(method='ada', tuneGrid=adagrid), #doesn't work at all
    svm=caretModelSpec(method='svmRadial',tuneGrid=svmgrid, preProcess=pp),
    #cforest=caretModelSpec(method='cforest',controls=cforcontrols), #this works, it just isn't included to have odd number
    #black=caretModelSpec(method='blackboost',tuneGrid=blackgrid,preProcess=pp), #something wrong, all ROC metric values are missing
    #earth=caretModelSpec(method='earth',tuneGrid=earthgrid, preProcess=pp),
    #gambo=caretModelSpec(method='gamboost', tuneGrid=gambogrid, preProcess=pp),
    logit=caretModelSpec(method='LogitBoost', tuneGrid=logitgrid, preProcess=pp) #whatch the comma I took out
    #bayesglm=caretModelSpec(method='bayesglm', preProcess=pp)
  )
)



###Wait HERE!!!!!!!!
# extract predicitons from this object for new data:
p <- as.data.frame(predict(model_list, newdata=head(testSet)))
print(p)
#No need to do the above, it's just a method to see the predictions of each model on new data

# WHAT MAKES A GOOD ENSEMBLE?
#Models w/ predicitons that are fairly un-correlated, but their overall accuaracy is similar.

#Lets take a closer look at our list of models:
xyplot(resamples(model_list))
# For each round of cross-validation, plot the area under the curve (ROC) as a scatterplot w/one model on x-axis, other on y
# The plot shows that the models are fairly un-correlated, the ROC values are almost never the same. 
# If the models were highly correlated, all of the points would fall perfectly on a single line

#confirm the 2 model’s correlation with the modelCor function from caret
modelCor(resamples(model_list)) #pearson's r correlation, scale 0->1. 0 is no corr, 1 is perfect. Corr can be + or - 
# These models are highly correlated. They'll probably suck

model_list$gbm #ROC = .86
summary(model_list$LogitBoost)

# BUILD A GREEDY ENSEMBLE
greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble) #See the overall AUC and the weights that each model was given
varImp(greedy_ensemble) #See the individual predictor ranking for the ensemble

#make predictions on the test set using the ensemble. This can be done my normal way, here's a new way
library('caTools')
model_preds <- lapply(model_list, predict, newdata=testing, type='prob')
model_preds <- lapply(model_preds, function(x) x[,'Survived']) # There are 2 outcomes Survived/Died. predict() returns probabilities for each outcome
# so predict()$Survived just gets the probability calcs for whether something is/isn't Survived. For each obs in testSet, generate a prob that
# it is Survived and that it is Died, vote/classify the observation as Survived or Died depending on which has a higher Prob.
model_preds <- data.frame(model_preds)
ens_preds <- predict(greedy_ensemble, newdata=testing)
model_preds$ensemble <- ens_preds
colAUC(model_preds, testing$Survived) #Calculate Area Under the ROC Curve (AUC) for every column of a matrix. Also, can be used to plot the ROC curves.
# greedy_ensemble had the highest overall prediction accuracy

# BUILD A STACKED MODEL
#caretStack allows us to move beyond simple blends of models to using “meta-models” to ensemble collections of predictive models.
# DO NOT use the trainControl object you used to fit the training models to fit the ensemble. The re-sampling indexes will be wrong
# Fortunately, you don’t need to be fastidious with re-sampling indexes for caretStack, as it only fits one model, and the defaults train uses will usually work fine:
glm_ensemble <- caretStack(
  model_list, 
  method='glm',
  metric='ROC',
  trControl=trainControl(
    method='boot',
    number=10,
    savePredictions=TRUE,
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)

model_preds2 <- model_preds
model_preds2$ensemble <- predict(glm_ensemble, newdata=testing, type='prob')$Survived
CF <- coef(glm_ensemble$ens_model$finalModel)[-1] #The [-1] just returns slopes for both glm and rpart. the [-1] removes the intercept
# CF just returns the coeffecients, the removal of the intercept coef is just that. The model was calc w/the intercept, CF just doesn't ask for it
# We remove the intercept, so that we can get a break down of the relative weights assigned to rpart & glm by the ensemble
colAUC(model_preds2, testing$Survived)
#Note that glm_ensemble$ens_model is a regular caret object of class train
CF/sum(CF) # This shows the weighting given to each model. These are very similar to those given by the greedy_ensemble

#use more sophisticated ensembles than simple linear weights
# these models are much more succeptible to over-fitting
# Non-linear ensembles seem to work best when you have:
# 1.Lots of data. large sets of resamples to train on (n=50 or higher for bootstrap samples).
# 2. Lots of models with similar accuracies.
# 3. Your models are un-correllated: each one seems to capture a different aspect of the data, and different models perform best on different subsets of the data.

library('gbm')
gbm_ensemble <- caretStack(
  model_list, 
  method='gbm',
  verbose=FALSE,
  tuneLength=10,
  metric='ROC',
  trControl=trainControl(
    method='boot',
    number=10,
    savePredictions=TRUE,
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)

model_preds3 <- model_preds
model_preds3$ensemble <- predict(gbm_ensemble, newdata=testing, type='prob')$Survived
colAUC(model_preds3, testing$Survived)
# In this case, the sophisticated ensemble did worse than a simple weighted linear combination
# But that's expected given elements 3 of what's necessary for a good non-linear ensemble are missing

#MAKE GREEDY PREDICTIONS ON testSet
testSet$Survived <- predict(greedy_ensemble, newdata=testSet)
#Generate SUBMISSION
library(dplyr)
greedyEnsSubmission <- testSet %>% select(PassengerId,Survived) #blendedSubmission <- subset(testSet, select= c(PassengerId,Survived))
# kaggle requires that the outcome is just (0,1)
greedyEnsSubmission1 <- greedyEnsSubmission %>% mutate(Survived = ifelse(Survived > 0.5, 1,0)) 
greedyEnsSubmission1 %>% group_by(Survived) %>% summarise(n()) #263 Died, 155 survived

#write resulting predictions w/only the two columns to csv
write.table(greedyEnsSubmission1,file = "greedyEnsSubmission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
# Same stupid score on Kaggle

#MAKE Stacked Linear PREDICTIONS ON testSet
testSet$Survived <- predict(glm_ensemble, newdata=testSet)
#Generate SUBMISSION
glmEnsSubmission <- testSet %>% select(PassengerId,Survived) #blendedSubmission <- subset(testSet, select= c(PassengerId,Survived))
# kaggle requires that the outcome is just (0,1)
glmEnsSubmission1 <- glmEnsSubmission %>% mutate(Survived = ifelse(Survived == "Survived", 1,0)) 
glmEnsSubmission1 %>% group_by(Survived) %>% summarise(n()) #264 Died, 154 survived

#write resulting predictions w/only the two columns to csv
write.table(glmEnsSubmission1,file = "glmEnsSubmission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
# Same stupid score

#MAKE Stacked GBM PREDICTIONS ON testSet
testSet$Survived <- predict(gbm_ensemble, newdata=testSet)
#Generate SUBMISSION
gbmEnsSubmission <- testSet %>% select(PassengerId,Survived) #blendedSubmission <- subset(testSet, select= c(PassengerId,Survived))
# kaggle requires that the outcome is just (0,1)
gbmEnsSubmission1 <- gbmEnsSubmission %>% mutate(Survived = ifelse(Survived == "Survived", 1,0)) 
gbmEnsSubmission1 %>% group_by(Survived) %>% summarise(n()) #285 Died, 133 survived

#write resulting predictions w/only the two columns to csv
write.table(gbmEnsSubmission1,file = "gbmEnsSubmission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
# Finally a different score: .75598 which is worse than any of my other attempts

