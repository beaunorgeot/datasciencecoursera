
setwd("/Users/bnorgeot/datasciencecoursera/titanic")

library(shiny)
source("simTitanic.R")

# #Features: Survived ~ Pclass + Sex + Fare + SibSp + Embarked + Parch
shinyServer(function(input, output) {
  
  output$didHeSurvive <- renderText({
    
    toTest <- data.frame(Sex=input$Sex, Pclass=input$Pclass, SibSp=input$SibSp, Fare=input$Fare, Embarked=input$Embarked, Parch=input$Parch)
    toTest$Pclass<-factor(toTest$Pclass, levels=c(1,2,3), labels=c("First class", "Second class", "Third class"))
    #
    Prediction <- predict(model, toTest)
    write.csv(Prediction, "prediction.csv")
    return(as.character(Prediction))
  })
})
