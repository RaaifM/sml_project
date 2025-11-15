
# 2095: Statistical and Machine Learning - Project 

# I) PRE-PROCESSING

set.seed("123")

# 1) Libraries 

library("tibble")
library("dplyr")
library("caret")
library("glmnet")
library("rpart")
library("rpart.plot")
library("randomForest")
library("MLmetrics")

# 2) Reading the data

# Getting location of the data file 

path <- file.path(here::here(""))
data_path <- file.path(path,"02_data","flight_data_2024.csv")
data <- read.csv(data_path)

# 3) Handling big data 
# Since the data is huge 7 million rows (1+ GB of data)
# We reduce the number of observations by removing some carriers

carrier_count <- table(data$op_unique_carrier)
print(carrier_count)

# In total we have 15 unique carriers 
# In this project we will only focus on the top 4 carriers by number of flights

# Sorting the above table 
sort_carrier_count <- sort(carrier_count, decreasing = TRUE)
top1_carriers <- head(sort_carrier_count, 1)
top1_codes <- names(top1_carriers)

# Here are the top 4 carriers: 
# WN = "Southwest Airlines", DL = "Delta Air Lines", 
# AA = "American Airlines", UA = "United Airlines"

# Filtering the data to only include top 1 carriers 

data_filtered <- data %>%
  filter(op_unique_carrier %in% top1_codes)

# 4) Removing redundant columns and after-even observed information columns 

drop_column_names <- c("fl_date","op_carrier_fl_num",
                       "origin_city_name","origin_state_nm",
                       "dest_city_name","dest_state_nm",
                       "crs_dep_time","dep_time","taxi_out",
                       "wheels_off","wheels_on","taxi_in",
                       "cancellation_code","actual_elapsed_time",
                       "air_time", "carrier_delay","weather_delay",
                       "nas_delay","security_delay","late_aircraft_delay",
                       "crs_arr_time","arr_time","year")

data_filtered <- data_filtered %>%
  select(-all_of(drop_column_names))

# 5) Constructing a new binary is_delayed column 

# Any flights with delay > 15, cancelled, or diverted is considered as delayed

data_filtered <- data_filtered %>%
  mutate(is_delayed = ifelse(cancelled ==1 | diverted == 1| coalesce(arr_delay > 15, FALSE), 1, 0))

# Let's get a first impression of how frequent delays are
table(data_filtered$is_delayed)

# 6) Dropping unnecessary columns (no longer needed)

data_filtered <- data_filtered %>%
  select(-c("cancelled","diverted","arr_delay"))

# 7) Make sure the data reads as factor where relevant 

# Converting the month, day_of_month, day_of_week, op_unique_carrier, origin, dest into factors

data_filtered$month <- factor(data_filtered$month)
# data_filtered$day_of_month <- factor(data_filtered$day_of_month)
data_filtered$day_of_week <- factor(data_filtered$day_of_week)
data_filtered$op_unique_carrier <- factor(data_filtered$op_unique_carrier)
data_filtered$origin <- factor(data_filtered$origin)
data_filtered$dest <- factor(data_filtered$dest)

# 8) If working with only 1-carrier data 

data_filtered <- data_filtered %>%
  select(-op_unique_carrier)

# 9) The data is way too big (we don't want to work with 1M + rows)

reduced_rows <- sample(nrow(data_filtered),20000)
data_filtered_reduced <- data_filtered[reduced_rows,]

# 10) drop all rows consisting of NA

data_filtered_reduced <- na.omit(data_filtered_reduced)
sum(is.na(data_filtered_reduced))

# 11) We also have to drop departure delay because it is highly correlated to is_delay
# We get warning in R for Quasi-complete separation 
data_filtered_reduced <- data_filtered_reduced %>% 
  select(-dep_delay)

# 12) Since factors like month, and day give rise to a lot of categorical variables we group them logically
# Handling the high-cardinality variables in the following way 

data_filtered_reduced$season <- factor(
  dplyr::case_when(
    data_filtered_reduced$month %in% c(12,1,2)  ~ "Winter",
    data_filtered_reduced$month %in% 3:5        ~ "Spring",
    data_filtered_reduced$month %in% 6:8        ~ "Summer",
    data_filtered_reduced$month %in% 9:11       ~ "Fall"
  )
)

data_filtered_reduced$day_group <- factor(
  dplyr::case_when(
    data_filtered_reduced$day_of_month <= 10 ~ "Early",
    data_filtered_reduced$day_of_month <= 20 ~ "Mid",
    TRUE                        ~ "Late"
  )
)

# Remove month, and day of month 
data_filtered_reduced <- data_filtered_reduced %>%
  select(-c("month","day_of_month","day_of_week"))

# II) SUITABLE PREDICTIVE MODELS

# 1) Logistic regression model (link = logit)

# Splitting the data into training and testing sets 

train_ind <- sample(nrow(data_filtered_reduced), size = 0.7*nrow(data_filtered_reduced))
train_l_data <- data_filtered_reduced[train_ind,]
test_l_data <- data_filtered_reduced[-train_ind,]

# Fitting the model 

logit_model <- glm(is_delayed ~ ., family = binomial(), data = data_filtered_reduced)

test_l_data$pred <- predict(logit_model, test_l_data, type = "response")
  
# Logistic Model Summary 

summary(logit_model)

# Prediction classes
test_l_data$pred_class <- ifelse(test_l_data$pred >=0.5,1,0)
  
# Ensure the predication and delay column are factors 
test_l_data$pred_class <- factor(test_l_data$pred_class , levels = c("0","1"))
test_l_data$is_delayed <- factor(test_l_data$is_delayed , levels = c("0","1"))

# Confusion matrix 
cm <- caret::confusionMatrix(data = test_l_data$pred_class, 
                       reference = test_l_data$is_delay, 
                       positive = "1")
acc <- cm$overall['Accuracy']
sens <- cm$byClass['Sensitivity']
spec <- cm$byClass['Specificity']
f1 <- cm$byClass['F1']

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# This is misleading for imbalanced data 
acc

# Sensitivity (Recall): TP / (TP + FN)
sens

# Specificity (True Negative Rate (TNR)): TN / (TN + FP)
spec

# F1
f1

# Significant factors 

# select rows with p-value < 0.05
coef_table <- summary(logit_model)$coefficients
signif_vars <- rownames(coef_table)[coef_table[,4] < 0.05]
signif_vars

# 2) Weighted Logistic model 

train_ind <- sample(nrow(data_filtered_reduced), size = 0.7*nrow(data_filtered_reduced))
train_logit_w_data <- data_filtered_reduced[train_ind,]
test_logit_w_data <- data_filtered_reduced[-train_ind,]

# We are dealing with an imbalanced data set
table(train_logit_w_data$is_delayed)/nrow(train_logit_w_data)

# About 80% of observation are not-delayed (0) and 20% are delayed (1)

# Creating class weights 

class_weights <- ifelse(train_logit_w_data$is_delayed == "1", 4, 1)
logit_weighted_model <- glm(is_delayed ~ ., 
                 data = train_logit_w_data, 
                 family = binomial(link = "logit"),
                 weights = class_weights)

test_logit_w_data$pred <- predict(logit_weighted_model, test_logit_w_data, type = "response")

# Logistic Weighted Model Summary 
summary(logit_weighted_model)

# Prediction classes
test_logit_w_data$pred_class <- ifelse(test_logit_w_data$pred >=0.5,1,0)

# Ensure the predication and delay column are factors 
test_logit_w_data$pred_class <- factor(test_logit_w_data$pred_class , levels = c("0","1"))
test_logit_w_data$is_delayed <- factor(test_logit_w_data$is_delayed , levels = c("0","1"))

# Confusion matrix 
cm <- caret::confusionMatrix(data = test_logit_w_data$pred_class, 
                             reference = test_logit_w_data$is_delay, 
                             positive = "1")
acc_logit_w <- cm$overall['Accuracy']
sens_logit_w <- cm$byClass['Sensitivity']
spec_logit_w <- cm$byClass['Specificity']
f1_logit_w <- cm$byClass['F1']

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# This is misleading for imbalanced data 
acc_logit_w

# Sensitivity (Recall): TP / (TP + FN)
sens_logit_w

# Specificity (True Negative Rate (TNR)): TN / (TN + FP)
spec_logit_w

# F1
f1_logit_w

# We get an over-all higher f1 score 

# 3) Weighted Lasso Regression Model 

train_ind <- sample(nrow(data_filtered_reduced), size = 0.7*nrow(data_filtered_reduced))
train_lasso_w_data <- data_filtered_reduced[train_ind,]
test_lasso_w_data <- data_filtered_reduced[-train_ind,]

class_weights <- ifelse(train_lasso_w_data$is_delayed == "1", 4, 1)

# Creating the design matrix (x_train) and response (y_train)
x_train <- model.matrix(is_delayed ~ ., data = train_lasso_w_data)[, -1]
y_train <- train_lasso_w_data$is_delay

# Hyper-parameter optimization: Running cross-validation to find the best Lambda 
cv_lasso_w_model <- cv.glmnet(x_train, y_train,
                            family = "binomial",
                            alpha = 1,
                            weights = class_weights)

# Plot 
plot(cv_lasso_w_model)

# Choosing the best lambda 
best_lambda <- cv_lasso_w_model$lambda.1se
coef(cv_lasso_w_model, s = best_lambda)

# All the coefficients with non-zero coefficients 
all_coefs_matrix <- as.matrix(coef(cv_lasso_w_model, s = best_lambda))
non_zero_coefs <- all_coefs_matrix[all_coefs_matrix[, 1] != 0,]
length(non_zero_coefs)

# Top 5 factors with the highest coefficients 
sorted_features <- non_zero_coefs[order(abs(non_zero_coefs), decreasing = TRUE)]
top_5_features <- head(sorted_features, 5)
top_5_features

# Making prediction using the cv_lasso_w_model 

x_test <- model.matrix(is_delayed ~ ., data = test_lasso_w_data)[, -1]
lasso_preds_class <- predict(cv_lasso_w_model, 
                             newx = x_test, 
                             s = best_lambda, 
                             type = "class")

factor_levels <- c("0", "1")

preds_factor <- factor(lasso_preds_class[, 1], levels = factor_levels)
ref_factor <- factor(test_lasso_w_data$is_delayed, levels = factor_levels)

# Confusion matrix 
cm_lasso <- caret::confusionMatrix(data = preds_factor, 
                                   reference = ref_factor, 
                                   positive = "1")

acc_lasso_w <- cm_lasso$overall['Accuracy']
sens_lasso_w <- cm_lasso$byClass['Sensitivity']
spec_lasso_w <- cm_lasso$byClass['Specificity']
f1_lasso_w <- cm_lasso$byClass['F1']

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# This is misleading for imbalanced data 
acc_lasso_w

# Sensitivity (Recall): TP / (TP + FN)
sens_lasso_w

# Specificity (True Negative Rate (TNR)): TN / (TN + FP)
spec_lasso_w

# F1
f1_lasso_w

# 4) Single Classification Tree - With all variables 

# Training and Testing data 
train_ind <- sample(nrow(data_filtered_reduced), size = 0.7*nrow(data_filtered_reduced))
train_tree_data <- data_filtered_reduced[train_ind,]
test_tree_data <- data_filtered_reduced[-train_ind,]

loss_matrix <- matrix(c(0, 1, 4, 0), byrow = TRUE, nrow = 2)

# a) Overfit tree (cp = 0.001)

full_tree <- rpart(is_delayed ~ ., 
                   data = train_tree_data, 
                   method = "class",
                   parms = list(loss = loss_matrix),
                   control = rpart.control(cp = 0.001)) 

printcp(full_tree)
plotcp(full_tree)

# Starting with a very small cp (complexity parameter) this 
# indicates the minimum amount of improvement that has to be done 
# for the new split to be added in the model

# b) Pruned-tree 

# Hyper-parameter optimization 
best_cp <- full_tree$cptable[which.min(full_tree$cptable[,"xerror"]), "CP"]
best_cp

# Now, we prune the tree using the best_cp
# Now, "prune" the tree to that best size
pruned_tree <- prune(full_tree, cp = best_cp)

rpart.plot(pruned_tree, 
           nn = TRUE,          
           extra = 104)

# Making predictions using the tree

tree_preds <- predict(pruned_tree, 
                      newdata = test_tree_data, 
                      type = "class") 
factor_levels <- c("0", "1")
preds_factor <- factor(tree_preds, levels = factor_levels)
ref_factor <- factor(test_tree_data$is_delayed, levels = factor_levels)

# Confusion matrix 
cm_tree <- caret::confusionMatrix(data = preds_factor, 
                                  reference = ref_factor, 
                                  positive = "1")
cm_tree

acc_tree<- cm_tree$overall['Accuracy']
sens_tree <- cm_tree$byClass['Sensitivity']
spec_tree <- cm_tree$byClass['Specificity']
f1_tree <- cm_tree$byClass['F1']

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# This is misleading for imbalanced data 
acc_tree

# Sensitivity Correctly predict delays (Recall): TP / (TP + FN)
sens_tree

# Specificity Correctly predict on-time (True Negative Rate (TNR)): TN / (TN + FP)
spec_tree

# F1
f1_tree

# 5) Single Classification tree - with lasso feature selection

# coefficients from best Lasso model
model_coefs <- coef(cv_lasso_w_model, s = best_lambda)

# Names of all variables with non-zero coefficients
non_zero_coefs_matrix <- as.matrix(model_coefs)
non_zero_vars_encoded <- rownames(non_zero_coefs_matrix)[non_zero_coefs_matrix[, 1] != 0]

# Remove intercept term from the list
non_zero_predictors_encoded <- non_zero_vars_encoded[non_zero_vars_encoded != "(Intercept)"]
print(non_zero_predictors_encoded)

# Map encoded names back to original variable names

# original predictor column names from your training data
original_predictors <- setdiff(colnames(train_lasso_w_data), "is_delayed")

selected_original_vars <- c()
for (var in original_predictors) {
  if (any(startsWith(non_zero_predictors_encoded, var))) {
    selected_original_vars <- c(selected_original_vars, var)
  }
}

# The lasso model drops the column of "day_group" and "crs_elapsed_time"
print(selected_original_vars)


# if at least one variable selected in "selected_original_vars"
if (length(selected_original_vars) > 0) {
  tree_formula_str <- paste("is_delayed ~", paste(selected_original_vars, collapse = " + "))
  tree_formula <- as.formula(tree_formula_str)
  
  print(paste("Building tree with formula:", tree_formula_str))
  
  tree_model_lasso_vars <- rpart(
    formula = tree_formula,
    data = train_lasso_w_data,
    method = "class",
    weights = class_weights # Re-using your weights
  )
  
  # 9. Display and plot the new, simpler tree
  print("Summary of the new tree (with Lasso variables):")
  printcp(tree_model_lasso_vars)
  
  rpart.plot(tree_model_lasso_vars, main = "Tree with Lasso-Selected Variables")
  
  tree_preds_lasso_vars <- predict(tree_model_lasso_vars, 
                                   newdata = test_lasso_w_data, 
                                   type = "class")
  
  tree_preds_factor <- factor(tree_preds_lasso_vars, levels = c("0", "1"))
  test_actual_factor <- factor(test_lasso_w_data$is_delayed, levels = c("0", "1"))
  
  cm_lasso_tree <- confusionMatrix(tree_preds_factor, test_actual_factor, positive = "1")
  
} else {
  print("Lasso selected no variables (lambda.1se might be too strict).")
  print("Try using cv_lasso_w_model$lambda.min instead.")
}


acc_lasso_tree<- cm_lasso_tree$overall['Accuracy']
sens_rf <- cm_rf$byClass['Sensitivity']
spec_rf <- cm_rf$byClass['Specificity']
f1_lasso_tree <- cm_lasso_tree$byClass['F1']

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# This is misleading for imbalanced data 
acc_rf

# Sensitivity Correctly predict delays (Recall): TP / (TP + FN)
sens_rf

# Specificity Correctly predict on-time (True Negative Rate (TNR)): TN / (TN + FP)
spec_rf

# F1
f1_rf



# 6) Random Forest 

train_ind <- sample(nrow(data_filtered_reduced), size = 0.7*nrow(data_filtered_reduced))
train_rf_data <- data_filtered_reduced[train_ind,]
test_rf_data <- data_filtered_reduced[-train_ind,]

train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  sampling = "down",      
  summaryFunction = prSummary, 
  classProbs = TRUE 
)

grid <- expand.grid(
  .mtry = c(5, 10, 20)
)

# Making sure the dependent isn't a number 
train_rf_data$is_delayed <- ifelse(train_rf_data$is_delayed == 1, "Delay", "NoDelay")
train_rf_data$is_delayed <- as.factor(train_rf_data$is_delayed)

# Hyper-parameter optimization for m_try 
rf_tuned_model <- train(
  is_delayed ~ .,
  data = train_rf_data,
  method = "rf",
  trControl = train_control,
  tuneGrid = grid,
  metric = "F1", 
  ntree = 100
)

# optimal m_try = 5
print(rf_tuned_model)
plot(rf_tuned_model)

# Predictions 
rf_preds <- predict(rf_tuned_model, newdata = test_rf_data)

# Re-converting back the dependent variable to binary
preds_numeric <- ifelse(rf_preds == "Delay", 1, 0)

factor_levels <- c("0", "1")

preds_factor <- factor(preds_numeric, levels = factor_levels)
ref_factor <- factor(test_rf_data$is_delayed, levels = factor_levels)

cm_rf <- caret::confusionMatrix(data = preds_factor, 
                                reference = ref_factor, 
                                positive = "1")

cm_rf

acc_rf<- cm_rf$overall['Accuracy']
sens_rf <- cm_rf$byClass['Sensitivity']
spec_rf <- cm_rf$byClass['Specificity']
f1_rf <- cm_rf$byClass['F1']

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# This is misleading for imbalanced data 
acc_rf

# Sensitivity Correctly predict delays (Recall): TP / (TP + FN)
sens_rf

# Specificity Correctly predict on-time (True Negative Rate (TNR)): TN / (TN + FP)
spec_rf

# F1
f1_rf

