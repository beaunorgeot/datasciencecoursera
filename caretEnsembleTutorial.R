
# caretEnsemble Tutorial: https://cran.r-project.org/web/packages/caretEnsemble/vignettes/caretEnsemble-intro.html
setwd("/Users/bnorgeot/datasciencecoursera")

# caretEnsemble has 3 primary functions: caretList, caretEnsemble and caretStack.
# 1. caretList is used to build lists of caret models on the same training data, with the same re-sampling parameters
# 2.  caretEnsemble and caretStack are used to create ensemble models from such lists of caret models.
# 3a. caretEnsemble uses greedy optimization to create a simple linear blend of models
# 3b. caretStack uses a caret model to combine the outputs from several component caret models.

#caretList is a flexible function for fitting many different caret models, with the same resampling parameters, to the same dataset.
#  It returns a convenient list of caret objects. has almost exactly the same arguments as train 

#caretEnsemble has 2 arguments that can be used to specify which models to fit: methodList and tuneList.
# methodList is a simple character vector of methods that will be fit with the default train parameters
# tuneList can be used to customize the call to each component model and will be discussed in more detail later

library('caret')
library('mlbench')
library('pROC')
data(Sonar)
set.seed(107)
inTrain <- createDataPartition(y = Sonar$Class, p = .75, list = FALSE)
training <- Sonar[ inTrain,]
testing <- Sonar[-inTrain,]
my_control <- trainControl(
  method='boot',
  number=25,
  savePredictions=TRUE,
  classProbs=TRUE,
  index=createResample(training$Class, 25),
  summaryFunction=twoClassSummary
)

# Notice that we are explicitly setting the resampling index to being used in trainControl. 
#If you do not set this index manually, caretList will attempt to set it for automatically, 
# but it’s generally a good idea to set it yourself.

#use caretList to fit a series of models (each with the same trControl):
library('rpart')
library('caretEnsemble')
model_list <- caretList(
  Class~., data=training,
  trControl=my_control,
  metric = "Accuracy", 
  methodList=c('glm', 'rpart')
)

# extract predicitons from this object for new data:
p <- as.data.frame(predict(model_list, newdata=head(testing)))
print(p)

#Experimenting________________
fitGLM <- train(Class ~., method = "glm", data= training, metric = "Accuracy") #results given are accuracy
fitGLMroc <- train(Class ~., method = "glm", data= training, metric = "roc") #results given are accuracy
fitGLMtr <- train(Class ~., method = "glm", data= training, metric = "Accuracy", trControl = my_control) #results given are ROC

my_control2 <- trainControl(
  method='repeatedcv',
  number=5,
  repeats=3,
  savePredictions=TRUE,
  classProbs=TRUE,
  summaryFunction=twoClassSummary
)

fitGLMtr2 <- train(Class ~., method = "glm", data= training, metric = "Accuracy", trControl = my_control2) #also gives ROC

my_control3 <- trainControl(
  method='repeatedcv',
  number=5,
  repeats=3,
  savePredictions=TRUE,
  classProbs=TRUE
)

fitGLMtr3 <- train(Class ~., method = "glm", data= training, metric = "Accuracy", trControl = my_control3) # Accuracy given. 
#so in trainControl, twoClassSummary provides ROC measurements and sensitivity/specificity while the defaultSummary gives accuracy. 
#To use twoClassSummary and/or mnLogLoss, the classProbs argument of trainControl should be TRUE.

#_______________end experimental____________________________________

# Get more control w/tuneList()
# This argumenent can be used to fit several different variants of the same model, 
# and can also be used to pass arguments through train down to the component functions
library('mlbench')
library('randomForest')
library('nnet')
model_list_big <- caretList(
  Class~., data=training,
  trControl=my_control,
  metric='ROC',
  methodList=c('glm', 'rpart'),
  tuneList=list(
    rf1=caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=2)),
    rf2=caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=10), preProcess='pca'),
    nn=caretModelSpec(method='nnet', tuneLength=2, trace=FALSE)
  )
)

# WHAT MAKES A GOOD ENSEMBLE?
#Models w/ predicitons that are fairly un-correlated, but their overall accuaracy is similar.

#Lets take a closer look at our list of models:
xyplot(resamples(model_list))
# For each round of cross-validation, plot the area under the curve (ROC) as a scatterplot w/one model on x-axis, other on y
# The plot shows that the models are fairly un-correlated, the ROC values are almost never the same. 
# If the models were highly correlated, all of the points would fall perfectly on a single line

#confirm the 2 model’s correlation with the modelCor function from caret
modelCor(resamples(model_list)) #pearson's r correlation, scale 0->1. 0 is no corr, 1 is perfect. Corr can be + or - 
# corr is .14, very low

model_list$glm #ROC = .6829
summary(model_list$glm) #can see all of the coeffeceints

# BUILD A GREEDY ENSEMBLE
greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble) #The ensemble outperforms either of the individuals on the training data
varImp(greedy_ensemble) #See the variable ranking for the ensemble
#make predictions on the test set using the ensemble. This can be done my normal way, here's a new way
library('caTools')
model_preds <- lapply(model_list, predict, newdata=testing, type='prob')
model_preds <- lapply(model_preds, function(x) x[,'M']) # There are 2 classes of sonar, M & R. predict() returns probabilities for each class
# so predict()$M just gets the probability calcs for whether something is/isn't M. For each obs in testSet, generate a prob that
# it is M and that it is R, vote/classify the observation as M or R depending on which has a higher Prob.
model_preds <- data.frame(model_preds)
ens_preds <- predict(greedy_ensemble, newdata=testing)
model_preds$ensemble <- ens_preds
colAUC(model_preds, testing$Class) #Calculate Area Under the ROC Curve (AUC) for every column of a matrix. Also, can be used to plot the ROC curves.

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
model_preds2$ensemble <- predict(glm_ensemble, newdata=testing, type='prob')$M
CF <- coef(glm_ensemble$ens_model$finalModel)[-1] #The [-1] just returns slopes for both glm and rpart. the [-1] removes the intercept
# CF just returns the coeffecients, the removal of the intercept coef is just that. The model was calc w/the intercept, CF just doesn't ask for it
# We remove the intercept, so that we can get a break down of the relative weights assigned to rpart & glm by the ensemble
colAUC(model_preds2, testing$Class)
#Note that glm_ensemble$ens_model is a regular caret object of class train
CF/sum(CF) # We see .318glm and .681rpart this is similar to the weights given by the greedy_ensemble

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
model_preds3$ensemble <- predict(gbm_ensemble, newdata=testing, type='prob')$M
colAUC(model_preds3, testing$Class)
# In this case, the sophisticated ensemble is no better than a simple weighted linear combination
# But that's expected given elements 1 & 3 of what's necessary for a good non-linear ensemble are missing
