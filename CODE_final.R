# FIRST STEPS

# Checking & installing missing packages
packages <- c("randomForest", "caret", "xgboost", "ROCR", "gbm", "compiler", "dplyr","ggpubr", "adabag", "scales", "factoextra", "vegan", "stringr", "clValid", "RColorBrewer", "DALEX", "MLmetrics")
install.packages(setdiff(packages, rownames(installed.packages())))
remove(packages)

# Load packages
library(randomForest)
library(caret)
library(xgboost)
library(ROCR)
library(gbm)
library(dplyr)
library(ggpubr)
library(adabag)
library(scales)
library(factoextra)
library(vegan)
library(stringr)
library(clValid)
library(RColorBrewer)
library(DALEX)
library(MLmetrics)

# Set working directory
setwd(getwd())

# Load data
data(data_final)

# Set species indices and sum
env_var_count <- 60 # ADJUST THIS ACCORDING TO DATA
spec_count <- 10 # ADJUST THIS ADJUST THIS ACCORDING TO DATA
spec_index <- "hrelab" # ("count", "indm2", "hrelab")
spec_sum <- "avg" # ("avg", "max")

# Apply data settings
if(spec_index=="count" & spec_sum=="max"){raw_data <- data.frame(data_final[,c(1:env_var_count,(env_var_count+1):(env_var_count+spec_count))])}
if(spec_index=="count" & spec_sum=="avg"){raw_data <- data.frame(data_final[,c(1:env_var_count,(env_var_count+1+(1*spec_count)):(env_var_count+spec_count+(1*spec_count)))])}
if(spec_index=="indm2" & spec_sum=="max"){raw_data <- data.frame(data_final[,c(1:env_var_count,(env_var_count+1+(2*spec_count)):(env_var_count+spec_count+(2*spec_count)))])}
if(spec_index=="indm2" & spec_sum=="avg"){raw_data <- data.frame(data_final[,c(1:env_var_count,(env_var_count+1+(3*spec_count)):(env_var_count+spec_count+(3*spec_count)))])}
if(spec_index=="hrelab" & spec_sum=="max"){raw_data <- data.frame(data_final[,c(1:env_var_count,(env_var_count+1+(4*spec_count)):(env_var_count+spec_count+(4*spec_count)))])}
if(spec_index=="hrelab" & spec_sum=="avg"){raw_data <- data.frame(data_final[,c(1:env_var_count,(env_var_count+1+(5*spec_count)):(env_var_count+spec_count+(5*spec_count)))])}

colnames(raw_data)[(env_var_count+1):(env_var_count+spec_count)] <- c("H.angu", "H.bulb", "H.bulg", "H.cont", "H.fulv", "H.inco", "H.inst", "H.mode", "H.pell", "H.saxo")

# Set basic values
seed <- 1
split_rate <- 0.75
method <- "cv"
metric <- "AUC" # ("AUC", "Accuracy")
hclust_method <- "ward.D" # ("single", "complete", "average", "ward.D1", "ward.D2", "mcquitty", "median", "centroid")
summaryFunction <- multiClassSummary
lvls_color <- c("#004488", "#117733", "#cc3311", "#835e4d", "#555555", "#ec7014", "#aa4499", "#ffff00", "#66ccee", "#aaaa00", "#000000")
lvls_color_pale_sp <- c("#ea512f", "#844324", "#737373", "#1e62a6", "#2f9551", "#ff8e32", "#c862b7", "#ffff1e", "#84eaff", "#c8c81e", "#1e1e1e")
species = colnames(raw_data[,(env_var_count+1):(env_var_count+spec_count)])

# STATISTICS OF ENVIRONMENTAL VARIABLES

env_var_stat <- data.frame("Variable"=colnames(raw_data[,1:env_var_count]))

for (i in c("min", "max", "mean", "sd")){
  env_var_stat <- cbind(env_var_stat, as.numeric(round(apply(raw_data[,1:env_var_count], 2, i, na.rm=TRUE),2)))
  colnames(env_var_stat)[ncol(env_var_stat)] <- paste0(i)
}

# SELECT THE MOST IMPORTANT VARIABLES PER SPECIES (PRES-ABS)

IMPORTANCE_varsel_plot <- data.frame()
ALE_plot <- data.frame()

for (i in c(1:length(species))){
  set.seed(seed)
  cvIndex <- createFolds(factor(data.frame("spec"=as.factor(ifelse(raw_data[,env_var_count+i]>0,"YES", "NO")), raw_data[,1:env_var_count])$spec), k = 5)
  rf_search_grid <- expand.grid(mtry= c(1: round(length(data.frame("spec"=as.factor(ifelse(raw_data[env_var_count+i]>0,"YES", "NO")))[,1][data.frame("spec"=as.factor(ifelse(raw_data[env_var_count+i]>0,"YES", "NO")))[,1]=="YES"])/10,0)))
  trControl_rf <- trainControl(index=cvIndex, method=method, search="grid", summaryFunction = summaryFunction,  classProbs = TRUE, savePredictions = "final")
  set.seed(seed)
  imp_data <- rfImpute(spec~ ., data= data.frame("spec"=as.factor(ifelse(raw_data[env_var_count+i]>0,"YES", "NO")), raw_data[,1:env_var_count]), iter=20, ntree=2000)
  set.seed(seed)
  rf.model <- train(spec~., data=imp_data, method="rf", metric=metric, trControl=trControl_rf, tuneGrid=rf_search_grid, verbose = FALSE)
  
  IMPORTANCE_varsel_plot <- rbind(IMPORTANCE_varsel_plot, data.frame("var"= rownames(head(data.frame(varImp(rf.model, scale = TRUE)$importance)[order(-data.frame(varImp(rf.model, scale = TRUE)$importance)$Overall),,drop=FALSE], ifelse(round(length(imp_data$spec[imp_data$spec=="YES"])/10,0)>3,3,round(length(imp_data$spec[imp_data$spec=="YES"])/10,0)))), "value"=head(data.frame(varImp(rf.model, scale = TRUE)$importance)[order(-data.frame(varImp(rf.model, scale = TRUE)$importance)$Overall),,drop=FALSE], ifelse(round(length(imp_data$spec[imp_data$spec=="YES"])/10,0)>3,3,round(length(imp_data$spec[imp_data$spec=="YES"])/10,0)))[,1], "Spec"=paste(species[i])))

  # Create ALE plot dataframe
  
  imp_data$spec2 <- as.numeric(ifelse(imp_data$spec=="YES",1,0))
  imp_data <- imp_data[,-1]
  names(imp_data)[names(imp_data) == "spec2"] <- "spec"
  explainer_rf.model <- DALEX::explain(model = rf.model, data = imp_data, y = imp_data$spec, label = "Random Forest")
  al_rf_FINAL <- model_profile(explainer = explainer_rf.model, type = "accumulated", variables = c(rownames(head(data.frame(varImp(rf.model, scale = TRUE)$importance)[order(-data.frame(varImp(rf.model, scale = TRUE)$importance)$Overall),,drop=FALSE], ifelse(round(length(imp_data$spec[imp_data$spec==1])/10,0)>3,3,round(length(imp_data$spec[imp_data$spec==1])/10,0))))))
  ALE_plot <- rbind(ALE_plot, data.frame("spec"=paste(species[i]), "var"=data.frame(al_rf_FINAL$agr_profiles)[,1], "x"=data.frame(al_rf_FINAL$agr_profiles)[,3], "y"=data.frame(al_rf_FINAL$agr_profiles)[,4]))
}

# Create ALE plots

  # Rescale and rename problematic values
  ALE_plot[ALE_plot$var=="coarse" | ALE_plot$var=="fine" | ALE_plot$var=="CLC_5",]$x <- as.numeric(ALE_plot[ALE_plot$var=="coarse" | ALE_plot$var=="fine" | ALE_plot$var=="CLC_5",]$x)*100
  ALE_plot[ALE_plot$var=="mvel_max" | ALE_plot$var=="mvel_min",]$x <- as.numeric(ALE_plot[ALE_plot$var=="mvel_max" | ALE_plot$var=="mvel_min",]$x)*100
  ALE_plot[ALE_plot$var=="coarse",]$var <- "coarse%"
  ALE_plot[ALE_plot$var=="fine",]$var <- "fine%"
  ALE_plot[ALE_plot$var=="CLC_5",]$var <- "CLC_5%"
  ALE_plot[ALE_plot$var=="mvel_max",]$var <- "mvel_max*"
  ALE_plot[ALE_plot$var=="mvel_min",]$var <- "mvel_min*"
  

ggarrange(
  ggplot(ALE_plot[ALE_plot$spec=="H.angu" | ALE_plot$spec=="H.bulb",], aes(x=x, y=y)) + 
    geom_line(size=0.8) + facet_grid(spec ~ var, scales = "free") + ylab("ALE") + 
    theme(plot.margin = margin(0.5,0.5,0.5,0.5, "cm"), axis.title.x=element_blank(), axis.text.x = element_text(size = 6), axis.text.y = element_text(size = 7), plot.title = element_text(hjust = 0.5, vjust=2), axis.title.y = element_text(size= 9, vjust = 3)) +
    scale_x_continuous(labels = scales::number_format(accuracy = 1, big.mark = "")) + 
    theme(strip.text.x = element_text(size = 8, margin = margin(0.02,0,0.1,0, "cm")), strip.text.y = element_text(size = 8, margin = margin(0,0.02,0,0.1, "cm"))),
  ggplot(ALE_plot[ALE_plot$spec=="H.inst" | ALE_plot$spec=="H.saxo" | ALE_plot$spec=="H.fulv",], aes(x=x, y=y)) + 
    geom_line(size=0.8) + facet_grid(spec ~ var, scales = "free") + ylab("ALE") + 
    theme(plot.margin = margin(0.5,0.5,0.5,0.5, "cm"), axis.title.x=element_blank(), axis.text.x = element_text(size = 6), axis.text.y = element_text(size = 7), plot.title = element_text(hjust = 0.5, vjust=2), axis.title.y = element_text(size= 9, vjust = 3)) +
    scale_x_continuous(labels = scales::number_format(accuracy = 1.0, big.mark = "")) +
    theme(strip.text.x = element_text(size = 8, margin = margin(0.02,0,0.1,0, "cm")), strip.text.y = element_text(size = 8, margin = margin(0,0.02,0,0.1, "cm"))),
  ggplot(ALE_plot[ALE_plot$spec=="H.pell" | ALE_plot$spec=="H.inco",], aes(x=x, y=y)) + 
    geom_line(size=0.8) + facet_grid(spec ~ var, scales = "free") + ylab("ALE") + 
    theme(plot.margin = margin(0.5,0.5,0.5,0.5, "cm"), axis.title.x=element_blank(), axis.text.x = element_text(size = 6), axis.text.y = element_text(size = 7), plot.title = element_text(hjust = 0.5, vjust=2), axis.title.y = element_text(size= 9, vjust = 3)) +
    scale_x_continuous(labels = scales::number_format(accuracy = 1.0, big.mark = "")) +
    theme(strip.text.x = element_text(size = 8, margin = margin(0.02,0,0.1,0, "cm")), strip.text.y = element_text(size = 8, margin = margin(0,0.02,0,0.1, "cm"))),
  ggplot(ALE_plot[ALE_plot$spec=="H.cont" | ALE_plot$spec=="H.mode" | ALE_plot$spec=="H.bulg",], aes(x=x, y=y)) + 
    geom_line(size=0.8) + facet_grid(spec ~ var, scales = "free") + ylab("ALE") + 
    theme(plot.margin = margin(0.5,0.5,0.5,0.5, "cm"), axis.title.x=element_blank(), axis.text.x = element_text(size = 6), axis.text.y = element_text(size = 7), plot.title = element_text(hjust = 0.5, vjust=2), axis.title.y = element_text(size= 9, vjust = 3)) +
    scale_x_continuous(labels = scales::number_format(accuracy = 1.0, big.mark = "")) +
    theme(strip.text.x = element_text(size = 8, margin = margin(0.02,0,0.1,0, "cm")), strip.text.y = element_text(size = 8, margin = margin(0,0.02,0,0.1, "cm"))),
labels=c("A", "B", "C", "D"), ncol = 2, nrow = 2)

IMP_vars <- data.frame("var"=unique(IMPORTANCE_varsel_plot$var))
IMP_vars <- cbind(IMP_vars, "first"=str_sub(IMP_vars$var, end=-4), "last_3"= ifelse(str_sub(IMP_vars$var, -3)=="min" | str_sub(IMP_vars$var, -3)=="max","avg", str_sub(IMP_vars$var, -3)))
IMP_vars <- cbind(IMP_vars, "final1"=paste(IMP_vars$first, IMP_vars$last_3, sep=""))
important_vars_list <- unique(IMP_vars$var)
var_check_list <- NULL

for (i in c(1:length(IMP_vars$final1))){
  var_check_list <- append(var_check_list, any(grepl(IMP_vars$final1[i], important_vars_list)))
}

IMP_vars <- cbind(IMP_vars, "check"=var_check_list)
IMP_vars <- cbind(IMP_vars, "FINAL"=ifelse(IMP_vars$check=="TRUE",IMP_vars$final1,IMP_vars$var))
imp_var_list <- unique(IMP_vars$FINAL)

remove(IMP_vars, important_vars_list, var_check_list, i, rf_search_grid, rf.model, imp_data, trControl_rf, cvIndex) # CLEAR

# CLUSTERING SPECIES TO GROUPS

df_spec <- scale(decostand(raw_data[, (env_var_count+1):(env_var_count+spec_count)], "hellinger"))
df_env <- raw_data[, which(colnames(raw_data[1:env_var_count]) %in% imp_var_list)]

cluster_this <- data.frame("Spec" = species, "RDA_1"=as.numeric(scores(rda(df_spec ~., df_env, na.action = na.omit, scaling = 1), choices = c(1,2,3,4), display="species")[,1]), "RDA_2"=as.numeric(scores(rda(df_spec ~., df_env, na.action = na.omit, scaling = 1), choices = c(1,2,3,4), display="species")[,2]), "RDA_3"=as.numeric(scores(rda(df_spec ~., df_env, na.action = na.omit, scaling = 1), choices = c(1,2,3,4), display="species")[,3]), "RDA_4"=as.numeric(scores(rda(df_spec ~., df_env, na.action = na.omit, scaling = 1), choices = c(1,2,3,4), display="species")[,4]))
rownames(cluster_this) <- cluster_this$Spec

#RDA plot

RDA_plot <- ggplot(cluster_this[,1:3], aes(x=RDA_1, y=RDA_2)) + 
  geom_point(size= 3, colour=ifelse(cluster_this[,1:3][1]=="H.bulg",lvls_color[4], ifelse(cluster_this[,1:3][1]=="H.saxo" | cluster_this[,1:3][1]=="H.fulv" | cluster_this[,1:3][1]=="H.inst", lvls_color[1], ifelse(cluster_this[,1:3][1]=="H.cont", lvls_color[5], ifelse(cluster_this[,1:3][1]=="H.inco" | cluster_this[,1:3][1]=="H.pell", lvls_color[2], ifelse(cluster_this[,1:3][1]=="H.mode", lvls_color[6], ifelse(cluster_this[,1:3][1]=="H.angu" | cluster_this[,1:3][1]=="H.bulb", lvls_color[3], "black"))))))) +
  geom_text(aes(label=rownames(cluster_this[,1:3])),size=4, vjust=0.5, hjust=-0.2) +
  geom_hline(yintercept=0, linetype="dotted") +
  geom_vline(xintercept=0, linetype="dotted") +
    theme(axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0)), axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), plot.title = element_text(hjust = 0.5, vjust=2), plot.margin = margin(0.5,0.5,0.5,0.5, "cm")) +
  ggtitle("RDA plot (Species abundance ~ selected important environmental variables)") +
  labs(x=paste("RDA1 (", round(as.numeric(summary(rda(df_spec ~., df_env, na.action = na.omit))$concont$importance[2,1])*100,2),"%)", sep=""), 
       y=paste("RDA2 (", round(as.numeric(summary(rda(df_spec ~., df_env, na.action = na.omit))$concont$importance[2,2])*100,2),"%)", sep=""))

plot(RDA_plot)

# Calculate Dunn-indices

cl_min = 4
cl_max = 6
dindex_list <- NULL
hclust_method_list <- c("single", "complete", "average", "ward.D1", "ward.D2", "mcquitty", "median", "centroid")

for (j in 1:length(hclust_method_list)){
  for (i in cl_min:cl_max) {
    cl <- as.numeric(cutree(hclust(dist(cluster_this[,2:5]), ifelse(hclust_method_list[j]== "ward.D1","ward.D", hclust_method_list[j])), k = i))
    cluster_this <- cbind(cluster_this, cl)
    colnames(cluster_this)[ncol(cluster_this)] <- paste("cl_",i,"_",hclust_method_list[j], sep="")
  }
}

dindex_calc_final <- data.frame()
dindex_calc_final <- data.frame("Cluster_nr"=c(cl_min:cl_max))

for (j in 1:length(hclust_method_list)){
  for (i in 1: ncol(cluster_this[,grepl(hclust_method_list[j],names(cluster_this))])){
    tmp <- dunn(dist(cluster_this[,2:5]),cluster_this[,grepl(hclust_method_list[j],names(cluster_this))][i])
    dindex_list <- append(dindex_list, tmp)
  }
  dindex_calc_final <- cbind(dindex_calc_final, dindex_list)
  colnames(dindex_calc_final)[ncol(dindex_calc_final)] <- hclust_method_list[j]
  dindex_list <- NULL
}

names(dindex_calc_final)[names(dindex_calc_final) == "ward.D1"] <- "ward.D"

species_cluster_nr <- dindex_calc_final$Cluster_nr[which.max(dindex_calc_final[,which(colnames(dindex_calc_final)==hclust_method)])]

species_cluster_nr <- species_cluster_nr # TWEAK THIS (species_custer_nr, 2:10)

cluster_this$Cluster <- cutree(hclust(dist(cluster_this[,2:5]), hclust_method), k = species_cluster_nr,)

spec_groups <- NULL

for (i in c(1:species_cluster_nr)){
  spec_groups <- append(spec_groups, paste(str_sub(cluster_this$Spec[cluster_this$Cluster==i], -4, -1), collapse="_"))
}

for (i in c(1:species_cluster_nr)){
  raw_data <- cbind(raw_data, rowSums(raw_data[,c(cluster_this$Spec[cluster_this$Cluster==i]), drop=FALSE]))
  colnames(raw_data)[ncol(raw_data)] <- spec_groups[i]
}

raw_data$HspecGr <- spec_groups[apply(raw_data[, c(names(raw_data)[(env_var_count+spec_count+1):(env_var_count+spec_count+length(spec_groups))])],1,which.max)]

# Cluster dendrogram

hc <- hclust(dist(cluster_this[,2:5]), hclust_method)
fviz_dend(hc, k = species_cluster_nr,cex = 0.8, lwd = 0.8, k_colors = lvls_color[1:species_cluster_nr],
          color_labels_by_k = TRUE,
          ggtheme = theme_classic(), rect=TRUE, rect_border = "grey40", rect_fill = TRUE, lower_rect = -0.45, horiz = TRUE, labels_track_height = 0) + 
          theme(axis.line.x = element_line(size = 0.5, colour = "black"), plot.title = element_text(hjust=0.5), axis.title.y = element_text(vjust = 3), plot.margin = margin(0.5,0.5,0.5,0.5, "cm"), axis.title.x = element_text(margin = margin(t=10, b=0, l=0, r=0))) + 
          labs(y = "Distance") + 
          ggtitle("Dendrogram of hierarchical clustering of species")

# STACKING

model_raw_data <- data.frame("HspecGr"= as.factor(raw_data$HspecGr), raw_data[,which(colnames(raw_data) %in% imp_var_list)])
set.seed(seed)
imp_raw_data <- rfImpute(HspecGr ~ ., data=model_raw_data, iter=20, ntree=2000)

# Creating BASIC_TRAIN and level2_test sets
set.seed(seed)
basic_train.index <- createDataPartition(imp_raw_data$HspecGr, p = split_rate, list = FALSE)
BASIC_TRAIN <- imp_raw_data[ basic_train.index,]
level2_test  <- imp_raw_data[-basic_train.index,]
remove(basic_train.index)
lvls = levels(BASIC_TRAIN$HspecGr)

set.seed(seed)
cvIndex <- createFolds(factor(BASIC_TRAIN$HspecGr), k=5)
METRICS_stack_plot <- data.frame()
IMPORTANCE_stack_plot <- data.frame()

# RF base learner
level1_rf_metrics <-NULL
rf_search_grid <- expand.grid(mtry= c(1:(ncol(BASIC_TRAIN)-1)))
trControl_rf <- trainControl(index=cvIndex, method=method, number=5, search="grid", summaryFunction = summaryFunction,  classProbs = TRUE, savePredictions = "final")
set.seed(seed)
level1.rf <- train(HspecGr~., data=BASIC_TRAIN, method="rf", metric=metric, trControl=trControl_rf, tuneGrid=rf_search_grid, verbose = FALSE)
level1.pred.rf <- predict(level1.rf, BASIC_TRAIN, type="raw")
level1.pred.prob.rf <- predict(level1.rf, BASIC_TRAIN, type="prob")
level1_rf_metrics <- append(level1_rf_metrics, as.numeric(level1.rf$results$Accuracy[level1.rf$results$mtry == as.numeric(level1.rf$bestTune)]))
level1_rf_metrics <- append(level1_rf_metrics, as.numeric(level1.rf$results$Kappa[level1.rf$results$mtry == as.numeric(level1.rf$bestTune)]))
level1_rf_metrics <- append(level1_rf_metrics, as.numeric(level1.rf$results$AUC[level1.rf$results$mtry == as.numeric(level1.rf$bestTune)]))

METRICS_stack <- data.frame("Acc"=level1_rf_metrics[1], "Kappa"=level1_rf_metrics[2], "AUC"=level1_rf_metrics[3], row.names="level1_rf")

METRICS_stack_plot <- rbind(METRICS_stack_plot, data.frame("value"=level1_rf_metrics, "model"="Random\nForest", "metric"=c("Accuracy", "Kappa", "AUC")))

# GBM base learner
level1_gbm_metrics <- NULL
gbm_search_grid <- expand.grid(n.trees = seq(from = 50, to = 1000, by = 50), interaction.depth = 5, shrinkage = c(0.1, 0.05, 0.025), n.minobsinnode = 2)
trControl_gbm <- trainControl(index=cvIndex, method=method, number=5, search="grid", summaryFunction = summaryFunction,  classProbs = TRUE, savePredictions = "final")
set.seed(seed)
level1.gbm <- train(HspecGr~., data=BASIC_TRAIN, method="gbm", metric=metric, trControl=trControl_gbm, tuneGrid=gbm_search_grid, verbose = FALSE)
level1.pred.gbm <- predict(level1.gbm, BASIC_TRAIN, type="raw")
level1.pred.prob.gbm <- predict(level1.gbm, BASIC_TRAIN, type="prob")
level1_gbm_metrics <- append(level1_gbm_metrics, level1.gbm$results$Accuracy[level1.gbm$results$shrinkage == level1.gbm$bestTune$shrinkage & level1.gbm$results$n.trees == level1.gbm$bestTune$n.trees])
level1_gbm_metrics <- append(level1_gbm_metrics, level1.gbm$results$Kappa[level1.gbm$results$shrinkage == level1.gbm$bestTune$shrinkage & level1.gbm$results$n.trees == level1.gbm$bestTune$n.trees])
level1_gbm_metrics <- append(level1_gbm_metrics, level1.gbm$results$AUC[level1.gbm$results$shrinkage == level1.gbm$bestTune$shrinkage & level1.gbm$results$n.trees == level1.gbm$bestTune$n.trees])

METRICS_stack <- rbind(METRICS_stack, "level1_gbm"=level1_gbm_metrics)

METRICS_stack_plot <- rbind(METRICS_stack_plot, data.frame("value"=level1_gbm_metrics, "model"="Generalized\nBoosting", "metric"=c("Accuracy", "Kappa", "AUC")))

#AdaBag base learner
level1_adabg_metrics <- NULL
adabg_search_grid <- expand.grid(mfinal = (1:3)*3, maxdepth = c(1, 3))
trControl_adabg <- trainControl(index=cvIndex, method=method, number=5, search="grid", summaryFunction = summaryFunction,  classProbs = TRUE, savePredictions = "final")
set.seed(seed)
level1.adabg <- train(HspecGr~., data=BASIC_TRAIN, method="AdaBag", metric=metric, trControl=trControl_adabg, tuneGrid=adabg_search_grid)
level1.pred.adabg <- predict(level1.adabg, BASIC_TRAIN, type="raw")
level1.pred.prob.adabg <- predict(level1.adabg, BASIC_TRAIN, type="prob")
level1_adabg_metrics <- append(level1_adabg_metrics, level1.adabg$results$Accuracy[level1.adabg$results$maxdepth == level1.adabg$bestTune$maxdepth & level1.adabg$results$mfinal == level1.adabg$bestTune$mfinal])
level1_adabg_metrics <- append(level1_adabg_metrics, level1.adabg$results$Kappa[level1.adabg$results$maxdepth == level1.adabg$bestTune$maxdepth & level1.adabg$results$mfinal == level1.adabg$bestTune$mfinal])
level1_adabg_metrics <- append(level1_adabg_metrics, level1.adabg$results$AUC[level1.adabg$results$maxdepth == level1.adabg$bestTune$maxdepth & level1.adabg$results$mfinal == level1.adabg$bestTune$mfinal])

METRICS_stack <- rbind(METRICS_stack, "level1_adabg"=level1_adabg_metrics)

METRICS_stack_plot <- rbind(METRICS_stack_plot, data.frame("value"=level1_adabg_metrics, "model"="Bagged\nAdaptive Boosting", "metric"=c("Accuracy", "Kappa", "AUC")))

# XGBoost base learner
level1_xgb_metrics <- NULL
xgb_search_grid <- expand.grid(nrounds = 100, eta = c(0.01, 0.1, 0.3), max_depth = c(2, 3, 5, 10), gamma = 0, colsample_bytree = 1, min_child_weight = 1, subsample = 1)
trControl_xgb <- trainControl(index=cvIndex, method=method, number=5, search="grid", summaryFunction = summaryFunction,  classProbs = TRUE, savePredictions = "final")
set.seed(seed)
level1.xgb <- train(HspecGr~., data=BASIC_TRAIN, method="xgbTree", metric=metric, trControl=trControl_xgb, tuneGrid=xgb_search_grid, verbose = FALSE)
level1.pred.xgb <- predict(level1.xgb, BASIC_TRAIN, type="raw")
level1.pred.prob.xgb <- predict(level1.xgb, BASIC_TRAIN, type="prob")
level1_xgb_metrics <- append(level1_xgb_metrics, level1.xgb$results$Accuracy[level1.xgb$results$eta == level1.xgb$bestTune$eta & level1.xgb$results$max_depth == level1.xgb$bestTune$max_depth])
level1_xgb_metrics <- append(level1_xgb_metrics, level1.xgb$results$Kappa[level1.xgb$results$eta == level1.xgb$bestTune$eta & level1.xgb$results$max_depth == level1.xgb$bestTune$max_depth])
level1_xgb_metrics <- append(level1_xgb_metrics, level1.xgb$results$AUC[level1.xgb$results$eta == level1.xgb$bestTune$eta & level1.xgb$results$max_depth == level1.xgb$bestTune$max_depth])

METRICS_stack <- rbind(METRICS_stack, "level1_xgb"=level1_xgb_metrics)

METRICS_stack_plot <- rbind(METRICS_stack_plot, data.frame("value"=level1_xgb_metrics, "model"="Extreme\nGradient Boosting", "metric"=c("Accuracy", "Kappa", "AUC")))

IMPORTANCE_stack_plot <- rbind(IMPORTANCE_stack_plot, data.frame("var"= rownames(head(data.frame(varImp(level1.rf, scale = TRUE)$importance)[order(-data.frame(varImp(level1.rf, scale = TRUE)$importance)$Overall),,drop=FALSE], 5)), "value"= as.numeric(head(data.frame(varImp(level1.rf, scale = TRUE)$importance)[order(-data.frame(varImp(level1.rf, scale = TRUE)$importance)$Overall),,drop=FALSE], 5)[,1]), "model"= "Random\nForest"), data.frame("var"= rownames(head(data.frame(varImp(level1.gbm, scale = TRUE)$importance)[order(-data.frame(varImp(level1.gbm, scale = TRUE)$importance)$Overall),,drop=FALSE], 5)), "value"= as.numeric(head(data.frame(varImp(level1.gbm, scale = TRUE)$importance)[order(-data.frame(varImp(level1.gbm, scale = TRUE)$importance)$Overall),,drop=FALSE], 5)[,1]), "model"="Generalized\nBoosting"), data.frame("var"= rownames(head(data.frame(varImp(level1.adabg, scale = TRUE)$importance)[order(-data.frame(varImp(level1.adabg, scale = TRUE)$importance)$Overall),,drop=FALSE], 5)), "value"= as.numeric(head(data.frame(varImp(level1.adabg, scale = TRUE)$importance)[order(-data.frame(varImp(level1.adabg, scale = TRUE)$importance)$Overall),,drop=FALSE], 5)[,1]), "model"="Bagged\nAdaptive\nBoosting"), data.frame("var"= rownames(head(data.frame(varImp(level1.xgb, scale = TRUE)$importance)[order(-data.frame(varImp(level1.xgb, scale = TRUE)$importance)$Overall),,drop=FALSE], 5)), "value"= as.numeric(head(data.frame(varImp(level1.xgb, scale = TRUE)$importance)[order(-data.frame(varImp(level1.xgb, scale = TRUE)$importance)$Overall),,drop=FALSE], 5)[,1]), "model"="Extreme\nGradient\nBoosting"))

imp_val_order <- aggregate(IMPORTANCE_stack_plot$value, list(IMPORTANCE_stack_plot$var), FUN=mean)[order(-aggregate(IMPORTANCE_stack_plot$value, list(IMPORTANCE_stack_plot$var), FUN=mean)[,2]),][,1]

imp_plot <- IMPORTANCE_stack_plot %>% 
  ggplot(aes(x = model, y = value, fill = reorder(var,value)), label=var) + 
  scale_fill_manual("", 
                    values= rev(brewer.pal(length(imp_val_order), "Blues")), 
                    breaks=imp_val_order, 
                    labels=imp_val_order) +
  ggtitle("Variable importance in base learner models") +
  labs(y="Importance % (scaled)") +
  theme(legend.position="none", axis.title.y = element_blank(), axis.title.x = element_text(margin = margin(t = 7.5, r = 0, b = 0, l = 0)), plot.title = element_text(hjust = 0.5, vjust=2), plot.margin = margin(0.5,0.5,0.5,0.5, "cm"))

for (i in levels(as.factor(IMPORTANCE_stack_plot$model))) {imp_plot <- imp_plot + geom_col(position = "dodge", width=0.8, data = IMPORTANCE_stack_plot %>% filter(model == i), colour="black")}

for (i in levels(as.factor(IMPORTANCE_stack_plot$model)))
{imp_plot <- imp_plot + geom_text(aes(y=1, label = var), hjust = 0, size = 3, position = position_dodge(width = 0.8), data = IMPORTANCE_stack_plot %>% filter(model == i), colour = ifelse(IMPORTANCE_stack_plot %>% filter(model == i) %>% select(var) == imp_val_order[1] | IMPORTANCE_stack_plot %>% filter(model == i) %>% select(var) == imp_val_order[2], "white", "black"))}

for (i in levels(as.factor(IMPORTANCE_stack_plot$model)))
{imp_plot <- imp_plot + geom_text(aes(label = round(value,2)), hjust = -0.2, size = 3, position = position_dodge(width = 0.8), data = IMPORTANCE_stack_plot %>% filter(model == i))}

imp_plot <- imp_plot + coord_flip()

# Create META_TRAIN set
META_TRAIN <- data.frame("Class"=BASIC_TRAIN$HspecGr, "RF_pred"=as.factor(level1.pred.rf))
META_TRAIN <- cbind(META_TRAIN, "gbm_pred"=as.factor(level1.pred.gbm))
META_TRAIN <- cbind(META_TRAIN, "adabg_pred"=as.factor(level1.pred.adabg))
META_TRAIN <- cbind(META_TRAIN, "xgb_pred"=as.factor(level1.pred.xgb))

# Level 2 predictions

#level2_rf
rf_search_grid <- expand.grid(mtry= level1.rf$bestTune$mtry)
trControl_rf <- trainControl(method="none", search="grid", summaryFunction = summaryFunction,  classProbs = TRUE, savePredictions = "final")
set.seed(seed)
level2.rf <- train(HspecGr~., data=BASIC_TRAIN, method="rf", metric=metric, trControl=trControl_rf, tuneGrid=rf_search_grid, verbose = FALSE)
level2.pred.rf <- predict(level2.rf, level2_test, type="raw")

#level2_gbm
gbm_search_grid <- expand.grid(n.trees = level1.gbm$bestTune$n.trees, interaction.depth = level1.gbm$bestTune$interaction.depth, shrinkage = level1.gbm$bestTune$shrinkage, n.minobsinnode = level1.gbm$bestTune$n.minobsinnode)
trControl_gbm <- trainControl(method="none", search="grid", summaryFunction = summaryFunction,  classProbs = TRUE, savePredictions = "final")
set.seed(seed)
level2.gbm <- train(HspecGr~., data=BASIC_TRAIN, method="gbm", metric=metric, trControl=trControl_gbm, tuneGrid=gbm_search_grid, verbose = FALSE)
level2.pred.gbm <- predict(level2.gbm, level2_test, type="raw")

#level2_adabg
adabg_search_grid <- expand.grid(mfinal = level1.adabg$bestTune$mfinal, maxdepth = level1.adabg$bestTune$maxdepth)
trControl_adabg <- trainControl(method="none", search="grid", summaryFunction = summaryFunction,  classProbs = TRUE, savePredictions = "final")
set.seed(seed)
level2.adabg <- train(HspecGr~., data=BASIC_TRAIN, method="AdaBag", metric=metric, trControl=trControl_adabg, tuneGrid=adabg_search_grid)
level2.pred.adabg <- predict(level2.adabg, level2_test, type="raw")

#level2_xgb
xgb_search_grid <- expand.grid(nrounds = level1.xgb$bestTune$nrounds, eta = level1.xgb$bestTune$eta, max_depth = level1.xgb$bestTune$max_depth, gamma = level1.xgb$bestTune$gamma, colsample_bytree = level1.xgb$bestTune$colsample_bytree, min_child_weight = level1.xgb$bestTune$min_child_weight, subsample = level1.xgb$bestTune$subsample)
trControl_xgb <- trainControl(method="none", search="grid", summaryFunction = summaryFunction,  classProbs = TRUE, savePredictions = "final")
set.seed(seed)
level2.xgb <- train(HspecGr~., data=BASIC_TRAIN, method="xgbTree", metric=metric, trControl=trControl_xgb, tuneGrid=xgb_search_grid, verbose = FALSE)
level2.pred.xgb <- predict(level2.xgb, level2_test, type="raw")

# Create META_TEST set
META_TEST <- data.frame("Class"=level2_test$HspecGr, "RF_pred"=as.factor(level2.pred.rf))
META_TEST <- cbind(META_TEST, "gbm_pred"=as.factor(level2.pred.gbm))
META_TEST <- cbind(META_TEST, "adabg_pred"=as.factor(level2.pred.adabg))
META_TEST <- cbind(META_TEST, "xgb_pred"=as.factor(level2.pred.xgb))

# META model
META_metrics <- NULL
set.seed(seed)
META.glm <- train(Class~., data= META_TRAIN, method="multinom", trace=FALSE)
META.pred.glm <- predict(META.glm, META_TEST, type="raw")
META.pred.prob.glm <- predict(META.glm, META_TEST, type="prob")

lvls = levels(BASIC_TRAIN$HspecGr)
aucs = c()
ROC_plot_final_GLM <- data.frame()
for (type.id in 1: length(lvls)) {
  type = as.factor(make.names(as.numeric(META_TRAIN$Class == lvls[type.id])))
  set.seed(seed)
  glmmodel = train(type~., data=cbind(META_TRAIN, "type"=type)[, -1], 
                   method="multinom", trace=FALSE)
  glmprediction = predict(glmmodel, META_TEST[,-1], type='prob')
  score = glmprediction[, 'X1']
  actual.class = META_TEST$Class == lvls[type.id]
  pred = prediction(score, actual.class)
  glmperf = performance(pred, "tpr", "fpr")
  roc.x = unlist(glmperf@x.values)
  roc.y = unlist(glmperf@y.values)
  #lines(roc.y ~ roc.x, col=lvls_color[type.id], lwd=3) 
  glmauc = performance(pred, "auc")
  glmauc = unlist(slot(glmauc, "y.values"))
  aucs[type.id] = glmauc
  ROC_plot_final_GLM <- rbind(ROC_plot_final_GLM, data.frame(roc.x,roc.y, Class=rep(lvls[type.id], length(roc.x))))
}

META_metrics <- append(META_metrics, as.numeric(confusionMatrix(table(META.pred.glm, META_TEST$Class))$overall[1]))
META_metrics <- append(META_metrics, as.numeric(confusionMatrix(table(META.pred.glm, META_TEST$Class))$overall[2]))
META_metrics <- append(META_metrics, mean(aucs))

METRICS_stack <- rbind(METRICS_stack, "META_model"=META_metrics)

METRICS_stack_plot <- rbind(METRICS_stack_plot, data.frame("value"=META_metrics, "model"="Meta-model\n(PMLR)", "metric"=c("Accuracy", "Kappa", "AUC")))

met_plot <- ggplot(METRICS_stack_plot, aes(x = reorder(model, value, mean), y=value, fill=factor(metric, levels=c("Kappa", "Accuracy", "AUC")))) + 
  geom_col(position="dodge", width=0.6, colour="black") + 
  scale_fill_manual("Metrics", values = c("#ec7014", "#cc3311", "#555555")) + 
  labs(y="Metric value") + 
  scale_y_continuous(breaks = seq(0, 1, by = .2), labels=label_number(accuracy=0.1), limits = c(-0.03,1)) + 
  ggtitle("Metrics of base learner models and the meta-model") + 
  theme(axis.title.x = element_blank(), plot.title = element_text(hjust = 0.5, vjust=2), axis.title.y = element_text(vjust = 3), plot.margin = margin(0.5,0.5,0.5,0.5, "cm"), legend.position = c(0.072,0.85)) +
  geom_text(aes(label = substring(as.character(format(round(value,3), nsmall=2)), 2)), size = 3, position = position_dodge(width = 0.6), vjust = 1.5, colour = ifelse(METRICS_stack_plot$metric=="AUC", "white", "black")) +
  annotate("text", label = "[mfinal, maxdepth]", x = 1, y = -0.03, colour="dimgrey", size=3.2, fontface="italic") +
  annotate("text", label = "[eta, maxdepth]", x = 2, y = -0.03, colour="dimgrey", size=3.2, fontface="italic") +
  annotate("text", label = "[n.trees, shrinkage]", x = 3, y = -0.03, colour="dimgrey", size=3.2, fontface="italic") +
  annotate("text", label = "[mtry]", x = 4, y = -0.03, colour="dimgrey", size=3.2, fontface="italic") +
  annotate("text", label = "[decay]", x = 5, y = -0.03, colour="dimgrey", size=3.2, fontface="italic")
  
confusionMatrix(META.pred.glm, META_TEST$Class)

ROC_plot <- ggplot(ROC_plot_final_GLM) + aes(x=roc.x, y=roc.y, group=Class, color=Class) + 
  geom_line(lwd=1.2) + geom_abline(lty=3) + 
  labs(x="False Positive Rate", y="True Positive Rate") + 
  ggtitle("Meta- (MNLR) model ROC curves") + 
  theme(plot.title = element_text(hjust = 0.5, vjust=2), axis.title.y = element_text(vjust = 3), axis.title.x = element_text(margin = margin(t = 7.5, r = 0, b = 0, l = 0)), plot.margin = margin(0.5,0.5,0.5,0.5, "cm"), legend.position = c(0.9,0.23)) + 
  scale_x_continuous(breaks=seq(0,1,by=0.2)) + scale_y_continuous(breaks=seq(0,1,by=0.2)) + scale_color_manual("Species groups", values=lvls_color_pale_sp)
  
plot(imp_plot)
plot(met_plot)
plot(ROC_plot)

# CLEAR UP ENVIRONMENT
remove(adabg_search_grid, al_rf_FINAL, explainer_rf.model, gbm_search_grid, rf_search_grid, trControl_adabg, trControl_gbm, trControl_rf, trControl_xgb, xgb_search_grid, actual.class, aucs, cl, dindex_list, glmauc, hclust_method_list, i, imp_val_order, j, level1_adabg_metrics, level1_gbm_metrics, level1_rf_metrics, level1_xgb_metrics, level1.pred.adabg, level1.pred.gbm, level1.pred.rf, level1.pred.xgb, level2.pred.adabg, level2.pred.gbm, level2.pred.rf, level2.pred.xgb, META_metrics, META.pred.glm, roc.x, roc.y, score, tmp, type, type.id)