# Spotify Playlist Analysis and User Prediction
#
# This script performs an analysis of Spotify playlist data, including audio features,
# and attempts to predict which user added a track to a collaborative playlist.
# Key steps include:
#   - Data loading from Spotify API
#   - Data validation and preprocessing
#   - Exploratory Data Analysis (EDA) via plots
#   - Logistic Regression modeling
#   - Handling imbalanced data using various techniques (Random Forest with rpart)
#   - Gradient Boosting Machine (GBM) modeling
#   - Feature engineering for date/time variables
#   - Final GBM model training and evaluation with new features

# --- 1. Load Libraries ---
library(spotifyr)    # For Spotify API interaction
library(tidyverse)   # For data manipulation (dplyr, ggplot2, purrr, etc.) and pipes
library(corrplot)    # For visualizing correlation matrices
library(caret)       # For model training and evaluation (includes createDataPartition, train, trainControl)
library(MASS)        # For stepAIC (stepwise model selection)
library(pROC)        # For ROC curve analysis
library(ROSE)        # For ROSE (Random Over-Sampling Examples) balancing technique
library(DMwR)        # For SMOTE (Synthetic Minority Over-sampling Technique) - Note: may need specific version or successor
library(gbm)         # For Gradient Boosting Machine models
library(lubridate)   # For date/time manipulation (ymd, interval, etc.)
library(viridis)     # For color scales in ggplot
library(rpart)       # For decision tree modeling (used with balancing techniques)

# --- 2. Configuration & Global Variables ---

# NOTE: These variables are expected to be defined before running the script.
# playlist_username <- "your_spotify_username"
# playlist_uris <- "spotify:playlist:your_playlist_id"

# Define class labels for logistic regression prediction output
POSITIVE_CLASS_LOGISTIC <- "LMM"
NEGATIVE_CLASS_LOGISTIC <- "kesarleon"

# Define target variable levels for GBM modeling (abbreviated for caret)
TARGET_LEVELS <- c('l', 'k')

# --- 3. Data Loading and Initial Preparation ---
message("Starting Data Loading and Initial Preparation...")

# Input validation for Spotify API credentials/playlist info
if (!exists("playlist_username") || is.null(playlist_username) || nchar(trimws(playlist_username)) == 0) {
  stop("Critical variable 'playlist_username' is not defined or is empty. Please define it before running the script.", call. = FALSE)
}
if (!exists("playlist_uris") || is.null(playlist_uris) || length(playlist_uris) == 0) {
  stop("Critical variable 'playlist_uris' is not defined or is empty. Please define it before running the script.", call. = FALSE)
}

# Fetch playlist audio features from Spotify API
message("Fetching playlist audio features from Spotify API...")
masterpiece <- tryCatch({
  get_playlist_audio_features(playlist_username, playlist_uris)
}, error = function(e) {
  message("Error fetching playlist features from Spotify API: ", e$message)
  stop("Stopping script due to API error.", call. = FALSE)
})
message("Playlist data fetched successfully.")

# Basic validation for the fetched 'masterpiece' dataframe
if (is.null(masterpiece)) {
  stop("The 'masterpiece' dataframe is NULL. Data loading may have failed catastrophically.", call. = FALSE)
}
if (!is.data.frame(masterpiece)) {
  stop("'masterpiece' is not a data frame. API might have returned unexpected structure.", call. = FALSE)
}
if (nrow(masterpiece) == 0) {
  stop("'masterpiece' dataframe is empty. No data to process from the playlist.", call. = FALSE)
}
required_cols_masterpiece <- c("added_by.id", "track.popularity", "track.album.release_date", "added_at")
if (!all(required_cols_masterpiece %in% names(masterpiece))) {
  missing_cols <- required_cols_masterpiece[!required_cols_masterpiece %in% names(masterpiece)]
  stop(paste("Missing essential columns in 'masterpiece':", paste(missing_cols, collapse=", ")), call. = FALSE)
}
message("Initial dataframe validation passed.")

# Define numerical features for initial analysis
num_cols <- c(
  "acousticness", "danceability", "energy", "instrumentalness", "key",
  "liveness", "loudness", "speechiness", "tempo", "time_signature",
  "track.duration_ms", "track.popularity", "valence"
)

# Select numerical features from the main dataframe
Num_master <- masterpiece %>%
  dplyr::select(all_of(num_cols))

# --- 4. Exploratory Data Analysis (EDA) - Initial ---
message("\nStarting Initial Exploratory Data Analysis...")

# Correlation plot of numerical features
message("Generating correlation plot of numerical features...")
if (ncol(Num_master) > 1 && sum(!sapply(Num_master, is.numeric)) == 0) {
  M <- cor(Num_master, use = "complete.obs")
  corrplot(M, method = "circle")
} else {
  message("Skipping correlation plot: Not enough numeric columns or non-numeric data present in Num_master.")
}

# Boxplot of numerical features before normalization
message("Displaying boxplot of numerical features (original scale)...")
boxplot(Num_master, main = "Boxplot of Numerical Features (Original)")

# Normalize numerical data (scaling)
message("Normalizing numerical data...")
Num_Data_Norm <- sapply(Num_master, scale)
message("Displaying boxplot of numerical features (normalized)...")
boxplot(Num_Data_Norm, main = "Boxplot of Numerical Features (Normalized)")
message("Summary of normalized numerical data:")
summary(Num_Data_Norm)


# --- 5. Logistic Regression Model ---
message("\nStarting Logistic Regression Modeling...")

# Prepare data for logistic regression
logist_master <- masterpiece %>%
  dplyr::select(all_of(num_cols), added_by.id) %>%
  mutate(added_by.id = as.factor(added_by.id))

# Initial logistic regression model
model_logistic_initial <- glm(added_by.id ~ ., family = binomial(link = 'logit'), data = logist_master)
message("Summary of the initial full logistic regression model:")
summary(model_logistic_initial)
message("ANOVA table for the initial full logistic regression model:")
anova(model_logistic_initial, test = "Chisq")

# Split data into training and testing sets
set.seed(123)
training.samples_logistic <- logist_master$added_by.id %>%
  createDataPartition(p = 0.7, list = FALSE)
train.data_logistic  <- logist_master[training.samples_logistic, ]
test.data_logistic <- logist_master[-training.samples_logistic, ]

if (!is.data.frame(train.data_logistic) || nrow(train.data_logistic) == 0) {
  stop("'train.data_logistic' is not a valid data frame or is empty.", call. = FALSE)
}
if (!is.data.frame(test.data_logistic) || nrow(test.data_logistic) == 0) {
  stop("'test.data_logistic' is not a valid data frame or is empty.", call. = FALSE)
}
message("Logistic regression data split into training and testing sets.")

# Fit logistic regression model with stepwise AIC selection
model_logistic_aic <- glm(added_by.id ~ ., data = train.data_logistic, family = binomial(link = 'logit')) %>%
  stepAIC(trace = FALSE)
message("Summary of logistic regression model after stepwise AIC selection:")
summary(model_logistic_aic)

# Predict probabilities on the test set
probabilities_logistic <- model_logistic_aic %>%
  predict(test.data_logistic, type = "response")
test.data_logistic$prediction <- probabilities_logistic

# Evaluate logistic model using ROC curve
message("ROC curve details for AIC-selected logistic regression model:")
roc_obj_logistic <- roc(test.data_logistic$added_by.id, test.data_logistic$prediction)
plot(roc_obj_logistic, main = "ROC Curve - Logistic Regression (AIC)")
print(roc_obj_logistic)
message("Optimal threshold for logistic regression (AIC) based on ROC (best method):")
coords(roc_obj_logistic, "best", "threshold", transpose = TRUE)

predicted.classes_logistic <- ifelse(probabilities_logistic > 0.88, POSITIVE_CLASS_LOGISTIC, NEGATIVE_CLASS_LOGISTIC)
accuracy_logistic <- mean(predicted.classes_logistic == test.data_logistic$added_by.id)
message(paste("Logistic Regression Accuracy on test data (Threshold 0.88):", round(accuracy_logistic, 4)))

message("Plotting sorted probabilities for logistic regression predictions...")
test.data_logistic %>%
  ggplot(aes(x = seq_along(prediction), y = sort(prediction), color = added_by.id)) +
  geom_point() +
  labs(title = "Logistic Regression: Sorted Probabilities by Class", x = "Index", y = "Sorted Predicted Probability") +
  theme_minimal() +
  theme(legend.position = 'bottom')

message("Summary of predicted probabilities (logistic regression):")
summary(probabilities_logistic)


# --- 6. Handling Imbalanced Data (with rpart Decision Trees) ---
message("\nExploring Techniques for Handling Imbalanced Data (using rpart)...")

message("Class distribution in original logistic training data:")
print(table(train.data_logistic$added_by.id))
message("Class distribution in original logistic testing data:")
print(table(test.data_logistic$added_by.id))

set.seed(123)
tree_imbalanced <- rpart(added_by.id ~ ., data = train.data_logistic)
pred_tree_imbalanced <- predict(tree_imbalanced, newdata = test.data_logistic)
message("Accuracy measures for rpart on imbalanced data:")
accuracy.meas(test.data_logistic$added_by.id, pred_tree_imbalanced[,2])
message("ROC curve for rpart on imbalanced data:")
roc.curve(test.data_logistic$added_by.id, pred_tree_imbalanced[,2], main = "ROC Curve (rpart, Imbalanced)")

message("Applying data balancing techniques...")
# a) Oversampling
data_balanced_over <- ovun.sample(added_by.id ~ ., data = train.data_logistic, method = "over", seed = 1)$data
message("Class distribution after Oversampling (training data):")
print(table(data_balanced_over$added_by.id))

# b) Undersampling
data_balanced_under <- ovun.sample(added_by.id ~ ., data = train.data_logistic, method = 'under', seed = 1)$data
message("Class distribution after Undersampling (training data):")
print(table(data_balanced_under$added_by.id))

# c) Both Over and Under Sampling
data_balanced_both <- ovun.sample(added_by.id ~ ., data = train.data_logistic, method = 'both', p = 0.5, seed = 1)$data
message("Class distribution after Both Over/Under (training data):")
print(table(data_balanced_both$added_by.id))

# d) ROSE
data_rose <- ROSE(added_by.id ~ ., data = train.data_logistic, seed = 1)$data
message("Class distribution after ROSE (training data):")
print(table(data_rose$added_by.id))

message("Training rpart models on balanced datasets...")
set.seed(123); tree_rose <- rpart(added_by.id ~ ., data = data_rose)
set.seed(123); tree_over <- rpart(added_by.id ~ ., data = data_balanced_over)
set.seed(123); tree_under <- rpart(added_by.id ~ ., data = data_balanced_under)
set.seed(123); tree_both <- rpart(added_by.id ~ ., data = data_balanced_both)

pred_tree_rose  <- predict(tree_rose,  newdata = test.data_logistic)
pred_tree_over  <- predict(tree_over,  newdata = test.data_logistic)
pred_tree_under <- predict(tree_under, newdata = test.data_logistic)
pred_tree_both  <- predict(tree_both,  newdata = test.data_logistic)

message("AUC for rpart with ROSE sampling on test data:")
roc.curve(test.data_logistic$added_by.id, pred_tree_rose[,2], main = "ROC (rpart, ROSE)")
message("AUC for rpart with Oversampling on test data:")
roc.curve(test.data_logistic$added_by.id, pred_tree_over[,2], main = "ROC (rpart, Oversampling)")
message("AUC for rpart with Undersampling on test data:")
roc.curve(test.data_logistic$added_by.id, pred_tree_under[,2], main = "ROC (rpart, Undersampling)")
message("AUC for rpart with Both (Over/Under) sampling on test data:")
roc.curve(test.data_logistic$added_by.id, pred_tree_both[,2], main = "ROC (rpart, Both)")

ROSE_holdout_eval <- ROSE.eval(added_by.id ~ ., data = train.data_logistic, learner = rpart, method.assess = "holdout", extr.pred = function(obj)obj[,2], seed = 1)
message("Results of ROSE evaluation with holdout (rpart):")
print(ROSE_holdout_eval)


# --- 7. Gradient Boosting Machine (GBM) - Initial Model ---
message("\nStarting Gradient Boosting Machine (GBM) Modeling - Initial Features...")
ctrl_gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = twoClassSummary, classProbs = TRUE)

levels(train.data_logistic$added_by.id) <- TARGET_LEVELS
levels(test.data_logistic$added_by.id)  <- TARGET_LEVELS

set.seed(1234)
gbm_orig_fit <- train(added_by.id ~ ., data = train.data_logistic, method = "gbm", verbose = FALSE, metric = "ROC", trControl = ctrl_gbm)
message("Summary of baseline GBM model (original data, initial features):")
print(gbm_orig_fit)
message("Plotting GBM performance for baseline model (ROC vs tuning parameters):")
plot(gbm_orig_fit, main="GBM Hyperparameter Tuning (Original Data, Initial Features)")

message("AUC for Baseline GBM on test data (initial features):")
roc(test.data_logistic$added_by.id, predict(gbm_orig_fit, test.data_logistic, type = "prob")[, TARGET_LEVELS[1]]) %>% auc() %>% print()

message("Exploring GBM with different data balancing strategies (initial features)...")
model_weights <- ifelse(train.data_logistic$added_by.id == TARGET_LEVELS[1], (1 / table(train.data_logistic$added_by.id)[TARGET_LEVELS[1]]) * 0.5, (1 / table(train.data_logistic$added_by.id)[TARGET_LEVELS[2]]) * 0.5)
ctrl_gbm_weighted <- ctrl_gbm; ctrl_gbm_weighted$seeds <- gbm_orig_fit$control$seeds
gbm_weighted_fit <- train(added_by.id ~ ., data = train.data_logistic, method = "gbm", verbose = FALSE, weights = model_weights, metric = "ROC", trControl = ctrl_gbm_weighted)
message("Summary of Weighted GBM model (initial features):")
print(gbm_weighted_fit)

ctrl_gbm_down <- ctrl_gbm; ctrl_gbm_down$sampling <- "down"; ctrl_gbm_down$seeds <- gbm_orig_fit$control$seeds
gbm_down_fit <- train(added_by.id ~ ., data = train.data_logistic, method = "gbm", verbose = FALSE, metric = "ROC", trControl = ctrl_gbm_down)
message("Summary of GBM with Down-sampling (initial features):")
print(gbm_down_fit)

ctrl_gbm_up <- ctrl_gbm; ctrl_gbm_up$sampling <- "up"; ctrl_gbm_up$seeds <- gbm_orig_fit$control$seeds
gbm_up_fit <- train(added_by.id ~ ., data = train.data_logistic, method = "gbm", verbose = FALSE, metric = "ROC", trControl = ctrl_gbm_up)
message("Summary of GBM with Up-sampling (initial features):")
print(gbm_up_fit)

ctrl_gbm_smote <- ctrl_gbm; ctrl_gbm_smote$sampling <- "smote"; ctrl_gbm_smote$seeds <- gbm_orig_fit$control$seeds
gbm_smote_fit <- train(added_by.id ~ ., data = train.data_logistic, method = "gbm", verbose = FALSE, metric = "ROC", trControl = ctrl_gbm_smote)
message("Summary of GBM with SMOTE (initial features):")
print(gbm_smote_fit)

gbm_model_list <- list(original = gbm_orig_fit, weighted = gbm_weighted_fit, down = gbm_down_fit, up = gbm_up_fit, SMOTE = gbm_smote_fit)
calculate_test_roc <- function(model, data, positive_class_level) { roc(data$added_by.id, predict(model, data, type = "prob")[, positive_class_level]) }
gbm_model_list_roc <- gbm_model_list %>% map(calculate_test_roc, data = test.data_logistic, positive_class_level = TARGET_LEVELS[1])
message("AUC values for different GBM models with various sampling techniques (initial features):")
gbm_model_list_roc %>% map(auc) %>% print()

results_list_roc_gbm <- list()
for(i in seq_along(gbm_model_list_roc)){ the_roc <- gbm_model_list_roc[[i]]; model_name <- names(gbm_model_list_roc)[i]; results_list_roc_gbm[[model_name]] <- tibble(tpr = the_roc$sensitivities, fpr = 1 - the_roc$specificities, model = model_name)}
results_df_roc_gbm <- bind_rows(results_list_roc_gbm)

message("Plotting ROC curves for GBM models with different sampling (initial features)...")
ggplot(aes(x = fpr,  y = tpr, group = model), data = results_df_roc_gbm) +
  geom_line(aes(color = model), size = 1) + scale_color_viridis(discrete = TRUE, option = 'D') +
  geom_abline(intercept = 0, slope = 1, color = "gray", linetype = "dashed", size = 1) +
  labs(title = "ROC Curves for GBM Models (Initial Features)", x = "False Positive Rate (1 - Specificity)", y = "True Positive Rate (Sensitivity)") +
  theme_minimal(base_size = 15) + theme(legend.position = "bottom")


# --- 8. Feature Engineering for Date/Time Variables ---
message("\nStarting Feature Engineering for Date/Time Variables...")
engineer_datetime_features <- function(df) {
  message("Engineering 'lanzamiento.track'...")
  lanzamiento <- ymd(df$track.album.release_date, quiet = TRUE)
  min_lanzamiento_global <- min(lanzamiento, na.rm = TRUE)
  if (is.infinite(min_lanzamiento_global)) {
      message("Warning: All 'track.album.release_date' are NA. 'lanzamiento.track' will be NA.")
      df$lanzamiento.track <- NA_real_
  } else {
      lanzamiento_interval <- interval(min_lanzamiento_global, lanzamiento)
      lanzamiento_days <- lanzamiento_interval %/% days(1)
      mean_lanzamiento_days <- mean(lanzamiento_days, na.rm = TRUE)
      if (is.nan(mean_lanzamiento_days)) mean_lanzamiento_days <- 0
      lanzamiento_days <- ifelse(is.na(lanzamiento_days), mean_lanzamiento_days, lanzamiento_days)
      df <- df %>% mutate(lanzamiento.track = lanzamiento_days)
  }

  message("Engineering 'added_at' derived features (trans, fecha, hora, dias)...")
  df <- df %>% mutate(added_at.trans = ymd_hms(added_at, quiet = TRUE), added_at.fecha = as_date(added_at.trans), added_at.hora = hour(added_at.trans))
  added_fecha <- df$added_at.fecha
  min_added_fecha_global <- min(added_fecha, na.rm = TRUE)
  if (is.infinite(min_added_fecha_global)) {
      message("Warning: All 'added_at.fecha' are NA. 'added_at.dias' will be NA.")
      df$added_at.dias <- NA_real_
  } else {
      added_interval <- interval(min_added_fecha_global, added_fecha)
      added_days <- added_interval %/% days(1)
      df <- df %>% mutate(added_at.dias = added_days)
  }
  return(df)
}

masterpiece <- engineer_datetime_features(masterpiece)
message("Feature engineering complete.")

message("Plotting EDA for new time-based features...")
if ("lanzamiento.track" %in% names(masterpiece)) {
  masterpiece %>% dplyr::select(track.id, lanzamiento.track, added_by.id) %>%
    ggplot(aes(x = seq_along(lanzamiento.track), y = sort(lanzamiento.track), color = added_by.id)) +
    geom_point(show.legend = FALSE) + labs(title = "Track Release Proximity ('lanzamiento.track')", x = "Index", y = "Sorted Days Since Earliest Release") + theme_minimal()
}
if ("added_at.hora" %in% names(masterpiece)) {
  masterpiece %>% dplyr::select(track.id, added_at.hora, added_by.id) %>%
    ggplot(aes(x = seq_along(added_at.hora), y = sort(added_at.hora), color = added_by.id)) +
    geom_point(show.legend = FALSE) + labs(title = "Hour of Day Track Added ('added_at.hora')", x = "Index", y = "Sorted Hour of Day Added") + theme_minimal()
}


# --- 9. Final Model with New Features (GBM) ---
message("\nStarting Final GBM Modeling with New Features...")
num_cols2 <- c("acousticness", "added_at.dias", "added_at.hora", "danceability", "energy", "instrumentalness", "lanzamiento.track", "liveness", "loudness", "speechiness", "tempo", "track.duration_ms", "track.popularity", "valence")
num_cols2_existing <- num_cols2[num_cols2 %in% names(masterpiece)]
if (length(num_cols2_existing) < length(num_cols2)){ message("Warning: Not all columns in num_cols2 are present in masterpiece. Using available subset: ", paste(num_cols2_existing, collapse=", "))}

Num_master_ft <- masterpiece %>% dplyr::select(all_of(num_cols2_existing))
message("Generating correlation plot for new feature set...")
if (ncol(Num_master_ft) > 1 && sum(!sapply(Num_master_ft, is.numeric)) == 0) { M_ft <- cor(Num_master_ft, use = "complete.obs"); corrplot(M_ft, method = "circle", main = "Correlation Plot (New Features)")} else { message("Skipping correlation plot for new features.")}

message("Displaying boxplots for new numerical features (original and normalized)...")
boxplot(Num_master_ft, main = "Boxplot of New Numerical Features (Original)")
Num_Data_Norm_ft <- sapply(Num_master_ft, scale)
boxplot(Num_Data_Norm_ft, main = "Boxplot of New Numerical Features (Normalized)")
message("Summary of normalized new numerical features:")
summary(Num_Data_Norm_ft)

logist_master_ft <- masterpiece %>% dplyr::select(all_of(num_cols2_existing), added_by.id) %>% mutate(added_by.id = as.factor(added_by.id))
set.seed(123); training.samples_ft <- logist_master_ft$added_by.id %>% createDataPartition(p = 0.7, list = FALSE)
train.data_ft  <- logist_master_ft[training.samples_ft, ]; test.data_ft <- logist_master_ft[-training.samples_ft, ]
if (!is.data.frame(train.data_ft) || nrow(train.data_ft) == 0) stop("'train.data_ft' is invalid.", call. = FALSE)
if (!is.data.frame(test.data_ft) || nrow(test.data_ft) == 0) stop("'test.data_ft' is invalid.", call. = FALSE)
message("New feature data split into training and testing sets.")

levels(train.data_ft$added_by.id) <- TARGET_LEVELS; levels(test.data_ft$added_by.id)  <- TARGET_LEVELS

set.seed(1234)
gbm_final_orig_fit <- train(added_by.id ~ ., data = train.data_ft, method = "gbm", verbose = FALSE, metric = "ROC", trControl = ctrl_gbm)
message("Summary of final GBM model (new features, original data):")
print(gbm_final_orig_fit)
message("Plotting final GBM performance (new features, original data):")
plot(gbm_final_orig_fit, main = "GBM Hyperparameter Tuning (New Features)")

message("AUC for Final GBM on test data (new features, original data):")
roc(test.data_ft$added_by.id, predict(gbm_final_orig_fit, test.data_ft, type = "prob")[, TARGET_LEVELS[1]]) %>% auc() %>% print()

model_weights_ft <- ifelse(train.data_ft$added_by.id == TARGET_LEVELS[1], (1 / table(train.data_ft$added_by.id)[TARGET_LEVELS[1]]) * 0.5, (1 / table(train.data_ft$added_by.id)[TARGET_LEVELS[2]]) * 0.5)
ctrl_gbm_ft_weighted <- ctrl_gbm; ctrl_gbm_ft_weighted$seeds <- gbm_final_orig_fit$control$seeds
gbm_final_weighted_fit <- train(added_by.id ~ ., data = train.data_ft, method = "gbm", verbose = FALSE, weights = model_weights_ft, metric = "ROC", trControl = ctrl_gbm_ft_weighted)
message("Summary of final Weighted GBM model (new features):")
print(gbm_final_weighted_fit)
message("Plotting final Weighted GBM performance (new features):")
plot(gbm_final_weighted_fit, main = "Weighted GBM Hyperparameter Tuning (New Features)")

message("AUC and ROC details for Final Weighted GBM (new features):")
pred_prob_final_weighted <- predict(gbm_final_weighted_fit, test.data_ft, type = 'prob')[, TARGET_LEVELS[1]]
roc_obj_final_weighted <- roc(test.data_ft$added_by.id, pred_prob_final_weighted)
print(auc(roc_obj_final_weighted))
plot(roc_obj_final_weighted, main = "ROC Curve - Final Weighted GBM (New Features)")
message("Optimal threshold for final weighted GBM (new features) based on ROC:")
coords(roc_obj_final_weighted, "best", "threshold", transpose = TRUE)

message("\n--- End of Script ---")
# Further steps could include:
# - More detailed hyperparameter tuning for GBM.
# - Trying other algorithms (e.g., Random Forest with ranger, XGBoost).
# - Deeper investigation of feature importance.
# - Cross-validation strategy refinement.
# - Saving model objects and predictions.
# - Generating a final report or presentation of findings.
