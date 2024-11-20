setwd("/Users/ronghaozeng/Desktop/AA/APAN 5200/PAC")
data = read.csv('analysis_data.csv',stringsAsFactors = T)
#set.seed(617)
library(skimr)
#skim(data)
library(mice)
data_complete= mice::complete(mice(data,method = 'rf',seed = 617))
skim(data_complete)
#split data into test and train
library(caret)
split = createDataPartition(y=data_complete$CTR,p = 0.7,list = F,groups = 100)
train = data_complete[split,]
test = data_complete[-split,]
#correlation understanding
numeric_data <- train[, sapply(train, is.numeric)]
library(tidyr); library(dplyr); library(ggplot2)
library(ggcorrplot)
ggcorrplot(cor(numeric_data[,-1]),
           method = 'square',
           type = 'lower',
           show.diag = F,
           colors = c('#e9a3c9', '#f7f7f7', '#a1d76a'))

#Stepwise Selection
start_mod = lm(CTR~1,data=train)
empty_mod = lm(CTR~1,data=train)
full_mod = lm(CTR~.,data=train)
hybridStepwise = step(start_mod,
                      scope=list(upper=full_mod,lower=empty_mod),
                      direction='both')
summary(hybridStepwise)

#basic model
model_basic <-lm(CTR ~ visual_appeal + targeting_score + ad_format + cta_strength + 
                   headline_length + contextual_relevance + device_type + brand_familiarity + 
                   market_saturation,data = train)
pred_test_basic<- predict(model_basic,newdata = test)
rmse_test_basic <- sqrt(mean((pred_test_basic-test$CTR)^2))
rmse_test_basic

#lasso with best lambda
# Set up cross-validation
library(glmnet)
trControl = trainControl(method = "cv", number = 5)
# Define the formula for the model
#formula <- CTR ~ targeting_score + visual_appeal + headline_length + cta_strength + body_word_count + #numeric 
  #position_on_page + ad_format + gender + location + time_of_day + day_of_week + device_type #factor
formula <- CTR ~.
# Train the model with 5-fold cross-validation
model_cv_lasso = train(formula, 
                 data = train, 
                 method = "glmnet", 
                 trControl = trControl, 
                 tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, by = 0.001)))
# Display cross-validation results
best_lambda <- model_cv_lasso$bestTune$lambda
print(best_lambda)
x = model.matrix(CTR~.-1,data=train)
y = train$CTR
model_lasso <- glmnet(x = x, y = y, alpha = 1, lambda = best_lambda)
x_test <- model.matrix(~ . - 1, data = test[, setdiff(names(test), "CTR")])
pred_test <- predict(model_lasso, newx = x_test, s = best_lambda)
rmse_lasso <- sqrt(mean((pred_test-test$CTR)^2))
print(rmse_lasso)

#Random Forest
library(randomForest)
trControl=trainControl(method="cv",number=5)
tuneGrid = expand.grid(mtry=1:ncol(train)-1)
cvModel = train(CTR~.,data=train,
                method="rf",ntree=100,trControl=trControl,tuneGrid=tuneGrid )
cvModel
# mtry = 25 has the best performance
cvForest = randomForest(CTR~.,data=train,ntree = 100,mtry=cvModel$bestTune$mtry)
pred_test_rf = predict(cvForest,newdata=test)
rmse_cv_forest = sqrt(mean((pred_test_rf-test$CTR)^2)); rmse_cv_forest
#0.06872009

#Ranger
library(ranger)
trControl=trainControl(method="cv",number=5,verboseIter = TRUE)                 # Display training progress)
tuneGrid = expand.grid(mtry=20:ncol(train)-1, 
                       splitrule = c('variance','extratrees','maxstat'), 
                       min.node.size = c(1,5,15,20)
                       )
cvModel_ranger = train(CTR~.,
                data=train,
                method="ranger",
                num.trees=1000,
                trControl=trControl,
                tuneGrid=tuneGrid )
cvModel_ranger
cv_forest_ranger = ranger(CTR~.,
                          data=train,
                          num.trees = 1000, 
                          mtry=cvModel_ranger$bestTune$mtry, 
                          min.node.size = cvModel_ranger$bestTune$min.node.size, 
                          splitrule = cvModel_ranger$bestTune$splitrule)
pred_test_ranger = predict(cv_forest_ranger, data =test, num.trees = 1000)
rmse_cv_forest_ranger = sqrt(mean((pred_test_ranger$predictions-test$CTR)^2)); rmse_cv_forest_ranger
#0.06690494
#The final values used for the model were mtry = 27, splitrule = variance and min.node.size= 15.

#boost
library(caret)
set.seed(617)
trControl = trainControl(method="cv",number=5)
tuneGrid = expand.grid(n.trees = 500, 
                       interaction.depth = c(1,2,3),
                       shrinkage = (1:100)*0.001,
                       n.minobsinnode=c(5,10,15))
garbage = capture.output(cvModel_boost <- train(CTR~.,
                                          data=train,
                                          method="gbm",
                                          trControl=trControl, 
                                          tuneGrid=tuneGrid))
 set.seed(617)
library(gbm)
cvboost = gbm(CTR~.,
              data=train,
              distribution="gaussian",
              n.trees=500,
              interaction.depth=cvModel_boost$bestTune$interaction.depth,
              shrinkage=cvModel_boost$bestTune$shrinkage,
              n.minobsinnode = cvModel_boost$bestTune$n.minobsinnode)
cvboost
pred_test_boost = predict(cvboost, newdata = test,n.trees=500)
rmse_test_boost = sqrt(mean((pred_test_boost - test$CTR)^2)); rmse_test_boost
#0.07250979

# Libraries
library(vtreat)
library(xgboost)
library(caret)

# Prepare the treatment plan excluding the target variable
trt <- designTreatmentsZ(dframe = train, varlist = setdiff(names(train), "CTR"))
newvars <- trt$scoreFrame[trt$scoreFrame$code %in% c('clean', 'lev'), 'varName']

# Prepare input matrices, excluding CTR
train_input <- prepare(treatmentplan = trt, dframe = train, varRestriction = newvars)
test_input <- prepare(treatmentplan = trt, dframe = test, varRestriction = newvars)

# Define a tuning grid
tune_grid <- expand.grid(
  nrounds =  c(200,1000),
  eta = c(0.005, 0.01, 0.05),         # Learning rate
  max_depth = c(3, 4, 5, 6),           # Tree depth
  gamma = c(0, 1),                  # Minimum loss reduction
  colsample_bytree = c(0.6, 0.7, 0.8),   # Column sampling ratio
  min_child_weight = c(1, 3),       # Minimum child weight
  subsample = c(0.6,0.7, 0.8)           # Row sampling ratio
)

# Cross-validation settings
trControl <- trainControl(
  method = "cv",                     # Cross-validation method
  number = 5,                        # Number of folds
  verboseIter = TRUE                 # Display training progress
)

# Train the XGBoost model using caret
xgb_model <- train(
  x = as.matrix(train_input),        # Features
  y = train$CTR,                     # Target variable
  method = "xgbTree",                # XGBoost method
  trControl = trControl,             # Cross-validation settings
  tuneGrid = tune_grid               # Tuning grid
)

# Display the best parameters
print(xgb_model$bestTune)

# Train the final model with the best parameters
final_xgb_model <- xgboost(
  data = xgb.DMatrix(as.matrix(train_input), label = train$CTR),
  nrounds = xgb_model$bestTune$nrounds,
  eta = xgb_model$bestTune$eta,
  max_depth = xgb_model$bestTune$max_depth,
  gamma = xgb_model$bestTune$gamma,
  colsample_bytree = xgb_model$bestTune$colsample_bytree,
  min_child_weight = xgb_model$bestTune$min_child_weight,
  subsample = xgb_model$bestTune$subsample,
  objective = "reg:squarederror",
  verbose = 0
)

# Predict on the test set
pred <- predict(final_xgb_model, newdata = as.matrix(test_input))

# Calculate RMSE
rmse_xgboost <- sqrt(mean((pred - test$CTR)^2))
print(rmse_xgboost)
#0.06699218

# Predict on scoring data
final_model<-cv_forest_ranger
scoring_data <- read.csv('scoring_data.csv',stringsAsFactors = T)
scoring_data_complete= mice::complete(mice(scoring_data,method = 'rf',seed = 617))
skim(scoring_data_complete)
# Fix factor level differences
for (col in names(train)) {
  if (is.factor(train[[col]]) && col %in% names(scoring_data_complete)) {
    scoring_data_complete[[col]] <- factor(scoring_data_complete[[col]], levels = levels(train[[col]]))
  }
}
#scoring_input <- prepare(treatmentplan = trt, dframe = scoring_data_complete, varRestriction = newvars)
pred_scoring_xgb <- predict(final_model, newdata = as.matrix(scoring_input))
pred_scoring_xgb
#pred_scoring_ranger<- predict(final_model, data = scoring_data_complete,num_trees=1000)
# Save the submission file
submission_file <- data.frame(id = scoring_data$id, CTR = pred_scoring_ranger$predictions)
write.csv(submission_file, 'sample_submission.csv', row.names = FALSE)
