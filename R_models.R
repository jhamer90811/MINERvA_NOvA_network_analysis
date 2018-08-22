# 10 - regressor lm/qm model; all accuracies; no transformations; normalized

data = read.csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_std_normalized.csv', row.names = 1)

data.lm = lm(final_accuracy~.-genealogy, data = data)
data.qm = lm(final_accuracy~(.-genealogy)^2, data = data)

# Diagnostic plots:

plot(data.lm)
plot(data.qm)

summary.lm = summary(data.lm)
summary.qm = summary(data.qm)

# Most significant regressors:
summary.lm$coefficients[order(summary.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.qm$coefficients[order(summary.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]

##############################

# 10 - regressor lm/qm model; acc > 0.054; y = log(acc + 1); normalized

data = read.csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_0.054_normed.csv', row.names = 1)

data$final_accuracy = log(data$final_accuracy + 1)

data.lm = lm(final_accuracy~.-genealogy, data = data)
data.qm = lm(final_accuracy~(.-genealogy)^2, data = data)

# Diagnostic plots:

plot(data.lm)
plot(data.qm)

summary.lm = summary(data.lm)
summary.qm = summary(data.qm)

# Most significant regressors:
summary.lm$coefficients[order(summary.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.qm$coefficients[order(summary.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]


##############################

# 32 - regressor lm/qm model; all accuracies; no transformations; normalized data

data = read.csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_all_acc_32_feat_normed.csv', row.names = 1)

data.lm = lm(final_accuracy~.-genealogy-initial_accuracy, data = data)
data.qm = lm(final_accuracy~(.-genealogy-initial_accuracy)^2, data = data)

# Diagnostic plots:

plot(data.lm)
plot(data.qm)

summary.lm = summary(data.lm)
summary.qm = summary(data.qm)

# Most significant regressors:
summary.lm$coefficients[order(summary.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.qm$coefficients[order(summary.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]


##############################

# 32 - regressor lm/qm model; acc> 0.054; y = log(acc + 1); normalized data

data = read.csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_0.054_all_features_normed.csv', row.names = 1)

data$final_accuracy = log(data$final_accuracy + 1)

data.lm = lm(final_accuracy~.-genealogy-initial_accuracy, data = data)
data.qm = lm(final_accuracy~(.-genealogy-initial_accuracy)^2, data = data)

# Diagnostic plots:

plot(data.lm)
plot(data.qm)

summary.lm = summary(data.lm)
summary.qm = summary(data.qm)

# Most significant regressors:
summary.lm$coefficients[order(summary.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.qm$coefficients[order(summary.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]


##############################

# 32 - regressor lm/qm model; acc> 0.054; y = log(acc + 1); normalized data; no intercept

data = read.csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_0.054_all_features_normed.csv', row.names = 1)

data$final_accuracy = log(data$final_accuracy + 1)

data.lm = lm(final_accuracy~.-genealogy-initial_accuracy-1, data = data)
data.qm = lm(final_accuracy~(.-genealogy-initial_accuracy)^2-1, data = data)

# Diagnostic plots:

plot(data.lm)
plot(data.qm)

summary.lm = summary(data.lm)
summary.qm = summary(data.qm)

# Most significant regressors:
summary.lm$coefficients[order(summary.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.qm$coefficients[order(summary.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]

#LM achieves R^2 = 0.04995
#QM achieves R^2 = 0.9169

##############################

# Take terms from previous lm/qm which are significant @ 0.001 level. (Change tol to analyze other
# models)

tol = 0.001

sig.terms.lm = row.names(summary.lm$coefficients[summary.lm$coefficients[,'Pr(>|t|)']<tol,])
sig.terms.qm = row.names(summary.qm$coefficients[summary.qm$coefficients[,'Pr(>|t|)']<tol,])

data$initial_accuracy = NULL
data$genealogy = NULL

#Note that the following formulae use the intercept term. 

formula.lm = as.formula(paste('final_accuracy', paste(sig.terms.lm, collapse=" + "), sep=" ~ "))
formula.qm = as.formula(paste('final_accuracy', paste(sig.terms.qm, collapse=" + "), sep=" ~ "))

parsimonious.lm = lm(formula.lm, data = data)
parsimonious.qm = lm(formula.qm, data = data)

# Diagnostic plots:

plot(parsimonious.lm)

plot(parsimonious.qm)

summary.parsimonious.lm = summary(parsimonious.lm)
summary.parsimonious.qm = summary(parsimonious.qm)

# Most significant regressors:
summary.parsimonious.lm$coefficients[order(summary.parsimonious.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.parsimonious.qm$coefficients[order(summary.parsimonious.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]

#The parsimonious lm w/intercept achieves R^2 = 0.3244 w/ all regressors still significant at 0.001 level

#The parsimonious qm w/intercept achieves R^2 = 0.389 w/ all but 10 regressors still significant at 0.001 level.

#####################################################################################################
#####################################################################################################


#Now for complex attribute models

# All regressor lm/qm model; all accuracies; no transformations; normalized

data = read.csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_all_normed.csv', row.names = 1)

data.lm = lm(final_accuracy~., data = data)
data.qm = lm(final_accuracy~(.)^2, data = data)

# Diagnostic plots:

plot(data.lm)
plot(data.qm)

summary.lm = summary(data.lm)
summary.qm = summary(data.qm)

# Most significant regressors:
summary.lm$coefficients[order(summary.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.qm$coefficients[order(summary.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]

#IN ORDER TO ISOLATE INDIVIDUAL SCORES, REPLACE "all" IN THE FILENAME BY THE SCORE ACRONYM (e.g. "acb", "flb2")

####################################################

# All regressor lm/qm model; accuracy > 0.055; y = log(acc + 1); normalized

data = read.csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_0.055_all_normed.csv', row.names = 1)

data$final_accuracy = log(data$final_accuracy + 1)

data.lm = lm(final_accuracy~., data = data)
data.qm = lm(final_accuracy~(.)^2, data = data)

# Diagnostic plots:

plot(data.lm)
plot(data.qm)

summary.lm = summary(data.lm)
summary.qm = summary(data.qm)

# Most significant regressors:
summary.lm$coefficients[order(summary.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.qm$coefficients[order(summary.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]


#IN ORDER TO ISOLATE INDIVIDUAL SCORES, REPLACE "all" IN THE FILENAME BY THE SCORE ACRONYM (e.g. "acb", "flb2")

######################################################

# 19-regressor lm/qm model; all_accuracy; no transformations; normalized

data = read.csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_normed_parsimonious.csv', row.names = 1)

data.lm = lm(final_accuracy~., data = data)
data.qm = lm(final_accuracy~(.)^2, data = data)

# Diagnostic plots:

plot(data.lm)
plot(data.qm)

summary.lm = summary(data.lm)
summary.qm = summary(data.qm)

# Most significant regressors:
summary.lm$coefficients[order(summary.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.qm$coefficients[order(summary.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]


######################################################

# 19-regressor lm/qm model; accuracy > 0.055; y = log(accuracy + 1); normalized

#(Note that the filename erroneously has a 0.1; the cutoff for accuracies in the dataset is actually 0.055)

data = read.csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_0.1_normed_parsimonious.csv', row.names = 1)

data$final_accuracy = log(data$final_accuracy + 1)

data.lm = lm(final_accuracy~., data = data)
data.qm = lm(final_accuracy~(.)^2, data = data)

# Diagnostic plots:

plot(data.lm)
plot(data.qm)

summary.lm = summary(data.lm)
summary.qm = summary(data.qm)

# Most significant regressors:
summary.lm$coefficients[order(summary.lm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]
summary.qm$coefficients[order(summary.qm$coefficients[,'Pr(>|t|)']),c('Estimate', 'Pr(>|t|)')]


######################################################

