library(tidyverse)
library(tidymodels)
library(kknn)
library(ranger)

##################################################################
# Which model predicts penguin `sex` best, from `bill_len` and `bill_dep`?
# Compare logistic regression, KNN, and random forest, all in tidymodels

##################################################################
# Step 0: training-testing split and cross-validation folds, shared by all models
# strata = sex keeps the female/male proportion the same in every split.
# The penguins are almost 50/50, so it matters little here, but use strata
# whenever the response is uneven (e.g. 10% vs 90%)
penguins_clean <- as_tibble(datasets::penguins) |> na.omit() |> select(-year)

set.seed(5)
penguins_split <- initial_split(penguins_clean, prop = 0.75, strata = sex)
penguins_train <- training(penguins_split)
penguins_test  <- testing(penguins_split)


##################################################################
# Step 1: specify the models
log_spec <- logistic_reg() |>
  set_engine("glm")

knn_spec <- nearest_neighbor(mode = "classification", neighbors = tune()) |>
  set_engine("kknn")

rf_spec <- rand_forest(mode = "classification", trees = 500, mtry = tune(), min_n = tune()) |>
  set_engine("ranger")

##################################################################
# Step 2: a recipe for each model
# logistic regression and random forest: no pre-processing needed
log_rec <- recipe(sex ~ bill_len + bill_dep, data = penguins_train)
rf_rec <- recipe(sex ~ bill_len + bill_dep, data = penguins_train)

# KNN: normalize so that distances are on the same scale
knn_rec <- recipe(sex ~ bill_len + bill_dep, data = penguins_train) |>
  step_normalize(all_numeric_predictors())

# check what a recipe does
knn_rec |> prep() |> bake(new_data = NULL) |> head(5)

##################################################################
# Step 3: a workflow for each model
log_wf <- workflow() |> add_model(log_spec) |> add_recipe(log_rec)
knn_wf <- workflow() |> add_model(knn_spec) |> add_recipe(knn_rec)
rf_wf  <- workflow() |> add_model(rf_spec)  |> add_recipe(rf_rec)

##################################################################
# Step 4: cross-validation, for all models

# Step 4a: the cross-validation folds, shared by all models
set.seed(5)
penguins_folds <- vfold_cv(penguins_train, v = 10, strata = sex)
penguins_folds

# Step 4b: the hyperparameter grid for each model (logistic regression: none)
knn_grid <- grid_regular(neighbors(range = c(1, 30)), levels = 30)
rf_grid <- grid_regular(mtry(range = c(1, 2)), min_n(range = c(2, 20)), levels = c(2, 4))
rf_grid

# Step 4c: fit each model on the folds, with the same metrics
cls_metrics <- metric_set(accuracy, roc_auc)
# save_pred = TRUE keeps the predictions on each validation fold, for the ROC curves
log_res <- fit_resamples(log_wf, resamples = penguins_folds, metrics = cls_metrics,  # nothing to tune
                         control = control_resamples(save_pred = TRUE))
knn_res <- tune_grid(knn_wf, resamples = penguins_folds, grid = knn_grid, metrics = cls_metrics,
                     control = control_grid(save_pred = TRUE))
set.seed(5)
rf_res  <- tune_grid(rf_wf, resamples = penguins_folds, grid = rf_grid, metrics = cls_metrics,
                     control = control_grid(save_pred = TRUE))

# Step 4d: average the metrics over the folds
collect_metrics(log_res)
collect_metrics(knn_res) |> filter(.metric == "accuracy") |> head(3)
autoplot(knn_res, metric = "accuracy")

# Step 4e: the best version of each model
show_best(knn_res, metric = "accuracy", n = 3)
show_best(rf_res, metric = "accuracy", n = 3)

##################################################################
# Step 4.5: compare the models on their cross-validation accuracy, and choose one
cv_compare <- bind_rows(
  collect_metrics(log_res) |> mutate(model = "Logistic regression"),
  show_best(knn_res, metric = "accuracy", n = 1) |> mutate(model = "KNN"),
  show_best(rf_res, metric = "accuracy", n = 1) |> mutate(model = "Random forest")
) |>
  filter(.metric == "accuracy") |>
  select(model, mean, std_err, n)
cv_compare

cv_compare |>
  ggplot(aes(x = mean, y = fct_reorder(model, mean))) +
  geom_pointrange(aes(xmin = mean - std_err, xmax = mean + std_err)) +
  labs(x = "Cross-validation accuracy, mean ± 1 standard error", y = NULL)

# compare the ROC curves of the best version of each model, on the validation folds
best_knn <- select_best(knn_res, metric = "accuracy")
best_rf <- select_best(rf_res, metric = "accuracy")

cv_preds <- bind_rows(
  collect_predictions(log_res) |> mutate(model = "Logistic regression"),
  collect_predictions(knn_res, parameters = best_knn) |> mutate(model = "KNN"),
  collect_predictions(rf_res, parameters = best_rf) |> mutate(model = "Random forest")
)

# male is the positive class: event_level = "second"
cv_preds |>
  group_by(model) |>
  roc_curve(truth = sex, .pred_male, event_level = "second") |>
  autoplot()

##################################################################
# Step 5/6: plug in the best k, refit the chosen model (KNN) on the training data,
# and evaluate it once on the testing data
best_k <- select_best(knn_res, metric = "accuracy")
knn_final <- last_fit(finalize_workflow(knn_wf, best_k), split = penguins_split, metrics = cls_metrics)
collect_metrics(knn_final)
