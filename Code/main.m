%% Telco Customer Churn - Decision Tree vs Naive Bayes 
clear; clc; close all;
% Force white background for all figures
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesColor', 'w');
set(groot, 'defaultAxesXColor', 'k');
set(groot, 'defaultAxesYColor', 'k');
set(groot, 'defaultTextColor', 'k');
rng(12345);

%% FIGURE SAVING SETTINGS
saveFigs = true;        
figDir = "Figures";
if saveFigs && ~exist(figDir,"dir")
    mkdir(figDir);
end

%% Load data
fname = "Telco-Customer-Churn.csv"; 
tbl = readtable(fname);

% Remove ID
if any(strcmpi(tbl.Properties.VariableNames,"customerID"))
    tbl.customerID = [];
end

% Convert TotalCharges
if any(strcmpi(tbl.Properties.VariableNames,"TotalCharges"))
    if iscell(tbl.TotalCharges) || isstring(tbl.TotalCharges)
        tmp = string(tbl.TotalCharges);
        tmp = strtrim(tmp);
        tmp(tmp=="") = "NaN";
        tbl.TotalCharges = double(tmp);
    end
end

% Convert target to categorical
tbl.Churn = categorical(tbl.Churn); 

% Convert any text predictors to categorical
varNames = tbl.Properties.VariableNames;
for i = 1:numel(varNames)
    if strcmp(varNames{i},"Churn"), continue; end
    v = tbl.(varNames{i});
    if iscell(v) || isstring(v)
        tbl.(varNames{i}) = categorical(v);
    end
end

% Remove missing rows 
tbl = rmmissing(tbl);

%% Basic EDA: target distribution
f1 = figure;
bar(countcats(tbl.Churn));
xticklabels(categories(tbl.Churn));
xlabel("Churn"); ylabel("Count");
title("Churn Distribution");
if saveFigs
    exportgraphics(f1, fullfile(figDir,"01_churn_distribution.png"), "Resolution", 300);
end

%% Feature engineering 
if any(strcmpi(varNames,"tenure")) && any(strcmpi(varNames,"TotalCharges"))
    tbl.AvgMonthlySpend = tbl.TotalCharges ./ max(tbl.tenure, 1);
end
if any(strcmpi(varNames,"Contract"))
    tbl.isLongTermContract = ismember(tbl.Contract, categorical(["One year","Two year"]));
end

%% Split train/test
cv = cvpartition(height(tbl), "HoldOut", 0.2);
trainTbl = tbl(training(cv), :);
testTbl  = tbl(test(cv), :);

Ytr = trainTbl.Churn;
Yte = testTbl.Churn;

predictorNames = setdiff(tbl.Properties.VariableNames, "Churn");


%% ==========================================================
%% BASELINE MODEL (Majority Class) - TEST SET
%% ==========================================================
% Majority class learned from TRAINING set
majorityClass = mode(Ytr);

% Predict majority class for all TEST samples
baselinePred = repmat(majorityClass, size(Yte));

% Baseline accuracy
baselineAcc = mean(baselinePred == Yte);

fprintf("Baseline (predict all '%s') Test Accuracy = %.3f\n", ...
    string(majorityClass), baselineAcc);

% Confusion matrix for baseline
fBase = figure;
confusionchart(Yte, baselinePred);
title("Baseline (Majority Class) - Confusion Matrix (Test Set)");

if saveFigs
    exportgraphics(fBase, fullfile(figDir,"00_baseline_confusion_test.png"), ...
        "Resolution", 300);
end


%% ---- Encoding: dummy variables for categoricals ----
[Xtr, featNames] = designMatrixFromTable(trainTbl(:, predictorNames));
[Xte, ~]         = designMatrixFromTable(testTbl(:, predictorNames), featNames);

%% ==========================================================
%% PART A) 5-FOLD CROSS-VALIDATION ON TRAINING SET 
%% ==========================================================
K = 5;
cvK = cvpartition(Ytr, "KFold", K);

accNB = zeros(K,1);
accTR = zeros(K,1);
aucNB = zeros(K,1);
aucTR = zeros(K,1);

for k = 1:K
    idxTrain = training(cvK, k);
    idxVal   = test(cvK, k);

    Xtr_k  = Xtr(idxTrain,:);
    Ytr_k  = Ytr(idxTrain);
    Xval_k = Xtr(idxVal,:);
    Yval_k = Ytr(idxVal);

    % Standardize INSIDE each fold for Naive Bayes (prevents leakage)
    [XtrZ_k, mu_k, sig_k] = zscore(Xtr_k);
    sig_k(sig_k==0) = 1;
    XvalZ_k = (Xval_k - mu_k) ./ sig_k;

    % --- Naive Bayes ---
    nb_k = fitcnb(XtrZ_k, Ytr_k);
    [predNB_k, scoreNB_k] = predict(nb_k, XvalZ_k);
    accNB(k) = mean(predNB_k == Yval_k);
    [~,~,~,aucNB(k)] = perfcurve(Yval_k, scoreNB_k(:,2), "Yes");

    % --- Decision Tree ---
    tree_k = fitctree(Xtr_k, Ytr_k, "MinLeafSize", 10);
    [predTR_k, scoreTR_k] = predict(tree_k, Xval_k);
    accTR(k) = mean(predTR_k == Yval_k);
    [~,~,~,aucTR(k)] = perfcurve(Yval_k, scoreTR_k(:,2), "Yes");
end

cvResults = table( ...
    ["Naive Bayes"; "Decision Tree"], ...
    [mean(accNB); mean(accTR)], ...
    [std(accNB);  std(accTR)], ...
    [mean(aucNB); mean(aucTR)], ...
    [std(aucNB);  std(aucTR)], ...
    'VariableNames', {'Model','MeanAccuracy','StdAccuracy','MeanAUC','StdAUC'});

disp("=== 5-Fold Cross-Validation Results (on training set) ===");
disp(cvResults);

if saveFigs
    writetable(cvResults, fullfile(figDir,"02_cv_results.csv"));
end

% Plot per-fold accuracies 
fCV = figure;
bar([accNB accTR]);
xlabel("Fold");
ylabel("Accuracy");
title("5-Fold Cross-Validation Accuracy per Fold");
legend("Naive Bayes","Decision Tree","Location","best");
grid on;

if saveFigs
    exportgraphics(fCV, fullfile(figDir,"03_cv_accuracy_per_fold.png"), "Resolution", 300);
end



%% ==========================================================
%% HYPERPARAMETER TUNING (Decision Tree) using CV on TRAINING SET
%% ==========================================================
% Small, coursework-friendly grid search (do NOT overdo it)
leafGrid  = [1 2 5 10 20 50];
splitGrid = [10 25 50 100];

tuningRows = [];

for ls = leafGrid
    for ms = splitGrid
        % CV estimates for this hyperparameter combo
        accFold = zeros(K,1);
        aucFold = zeros(K,1);

        for k = 1:K
            idxTrain = training(cvK, k);
            idxVal   = test(cvK, k);

            Xtr_k  = Xtr(idxTrain,:);
            Ytr_k  = Ytr(idxTrain);
            Xval_k = Xtr(idxVal,:);
            Yval_k = Ytr(idxVal);

            tree_k = fitctree(Xtr_k, Ytr_k, "MinLeafSize", ls, "MaxNumSplits", ms);
            [pred_k, score_k] = predict(tree_k, Xval_k);

            accFold(k) = mean(pred_k == Yval_k);
            [~,~,~,aucFold(k)] = perfcurve(Yval_k, score_k(:,2), "Yes");
        end

        tuningRows = [tuningRows; ...
            ls, ms, mean(accFold), std(accFold), mean(aucFold), std(aucFold)];
    end
end

tuningResults = array2table(tuningRows, ...
    'VariableNames', {'MinLeafSize','MaxNumSplits','MeanAcc','StdAcc','MeanAUC','StdAUC'});

% Pick the best model by MeanAUC first, then MeanAcc
tuningResults = sortrows(tuningResults, {'MeanAUC','MeanAcc'}, {'descend','descend'});
bestParams = tuningResults(1,:);

disp("=== Decision Tree Hyperparameter Tuning Results (Top 10) ===");
disp(tuningResults(1:min(10,height(tuningResults)),:));

disp("=== Best Decision Tree Hyperparameters (by CV MeanAUC) ===");
disp(bestParams);

if saveFigs
    writetable(tuningResults, fullfile(figDir,"09_tree_hyperparameter_tuning_results.csv"));
end

% Plot: MeanAUC as a heatmap-like matrix (MaxNumSplits x MinLeafSize)
aucMat = NaN(numel(splitGrid), numel(leafGrid));
for i = 1:height(tuningResults)
    ls = tuningResults.MinLeafSize(i);
    ms = tuningResults.MaxNumSplits(i);
    r = find(splitGrid==ms);
    c = find(leafGrid==ls);
    aucMat(r,c) = tuningResults.MeanAUC(i);
end

fTune = figure;
imagesc(aucMat);
colorbar;
title("Decision Tree Tuning (CV Mean AUC)");
xlabel("MinLeafSize");
ylabel("MaxNumSplits");
xticks(1:numel(leafGrid)); xticklabels(string(leafGrid));
yticks(1:numel(splitGrid)); yticklabels(string(splitGrid));

if saveFigs
    exportgraphics(fTune, fullfile(figDir,"10_tree_tuning_heatmap_auc.png"), "Resolution", 300);
end




%% ==========================================================
%% FINAL TRAINING ON FULL TRAIN SET + TEST SET EVALUATION
%% ==========================================================

%% Standardization (Naive Bayes) using TRAINING stats only
[XtrZ, mu, sig] = zscore(Xtr);
sig(sig==0) = 1;
XteZ = (Xte - mu) ./ sig;

%% Model 1: Naive Bayes (Gaussian NB) - final
nb = fitcnb(XtrZ, Ytr);
[nbPred, nbScore] = predict(nb, XteZ);

%% Model 2: Decision Tree - final
% Use tuned hyperparameters from CV
tree = fitctree(Xtr, Ytr, ...
    "MinLeafSize", bestParams.MinLeafSize, ...
    "MaxNumSplits", bestParams.MaxNumSplits);
[treePred, treeScore] = predict(tree, Xte);

%% Final evaluation (prints metrics)
disp("=== FINAL TEST SET RESULTS ===");

disp("Naive Bayes:");
showMetrics(Yte, nbPred, nbScore, "Naive Bayes (Test)");

disp("Decision Tree:");
showMetrics(Yte, treePred, treeScore, "Decision Tree (Test)");

%% Confusion Matrices 
f2 = figure;
confusionchart(Yte, nbPred);
title("Naive Bayes - Confusion Matrix (Test Set)");
if saveFigs
    exportgraphics(f2, fullfile(figDir,"04_confusion_naive_bayes_test.png"), "Resolution", 300);
end

f3 = figure;
confusionchart(Yte, treePred);
title("Decision Tree - Confusion Matrix (Test Set)");
if saveFigs
    exportgraphics(f3, fullfile(figDir,"05_confusion_decision_tree_test.png"), "Resolution", 300);
end

%% ROC curves 
posClass = "Yes";
[XB, YB, ~, AUC_nb] = perfcurve(Yte, nbScore(:,2), posClass);
[XT, YT, ~, AUC_tr] = perfcurve(Yte, treeScore(:,2), posClass);

f4 = figure;
plot(XB, YB, "-"); hold on;
plot(XT, YT, "--");
xlabel("False Positive Rate"); ylabel("True Positive Rate");
title("ROC Curves (Test Set)");
legend("Naive Bayes (AUC=" + string(round(AUC_nb,3)) + ")", ...
       "Decision Tree (AUC=" + string(round(AUC_tr,3)) + ")", ...
       "Location","SouthEast");
grid on;

if saveFigs
    exportgraphics(f4, fullfile(figDir,"06_roc_curves_test.png"), "Resolution", 300);
end



%% Precision-Recall Curves (Test Set)
[posX_nb, posY_nb, ~, AUPRC_nb] = perfcurve(Yte, nbScore(:,2), "Yes", "xCrit","reca", "yCrit","prec");
[posX_tr, posY_tr, ~, AUPRC_tr] = perfcurve(Yte, treeScore(:,2), "Yes", "xCrit","reca", "yCrit","prec");

fPR = figure;
plot(posX_nb, posY_nb, "-"); hold on;
plot(posX_tr, posY_tr, "--");
xlabel("Recall"); ylabel("Precision");
title("Precision-Recall Curves (Test Set)");
legend("Naive Bayes (AUPRC=" + string(round(AUPRC_nb,3)) + ")", ...
       "Decision Tree (AUPRC=" + string(round(AUPRC_tr,3)) + ")", ...
       "Location","best");
grid on;

if saveFigs
    exportgraphics(fPR, fullfile(figDir,"13_precision_recall_curves_test.png"), "Resolution", 300);
end


%% ==========================================================
%% FINAL COMPARISON TABLE: CV vs TEST (Naive Bayes vs Tuned Decision Tree)
%% ==========================================================
% Compute TEST metrics into tables
testNB = computeMetricsTable(Yte, nbPred, nbScore, "Naive Bayes", AUC_nb);
testTR = computeMetricsTable(Yte, treePred, treeScore, "Decision Tree (Tuned)", AUC_tr);

% Pull CV means/stds for NB and Tree(Tuned)
% NB CV stats from earlier (accNB/aucNB arrays)
cvNB_meanAcc = mean(accNB);  cvNB_stdAcc = std(accNB);
cvNB_meanAUC = mean(aucNB);  cvNB_stdAUC = std(aucNB);

% Tree tuned CV stats from tuningResults/bestParams
cvTR_meanAcc = bestParams.MeanAcc;
cvTR_stdAcc  = bestParams.StdAcc;
cvTR_meanAUC = bestParams.MeanAUC;
cvTR_stdAUC  = bestParams.StdAUC;

finalComparison = table( ...
    ["Naive Bayes"; "Decision Tree (Tuned)"], ...
    [cvNB_meanAcc; cvTR_meanAcc], ...
    [cvNB_stdAcc;  cvTR_stdAcc], ...
    [cvNB_meanAUC; cvTR_meanAUC], ...
    [cvNB_stdAUC;  cvTR_stdAUC], ...
    [testNB.Accuracy; testTR.Accuracy], ...
    [testNB.F1;       testTR.F1], ...
    [testNB.AUC;      testTR.AUC], ...
    'VariableNames', {'Model','CV_MeanAcc','CV_StdAcc','CV_MeanAUC','CV_StdAUC','Test_Accuracy','Test_F1','Test_AUC'});

disp("=== FINAL COMPARISON (CV vs TEST) ===");
disp(finalComparison);

if saveFigs
    writetable(finalComparison, fullfile(figDir,"11_final_comparison_cv_vs_test.csv"));
end

% Optional plot: compare CV mean AUC vs Test AUC
fComp = figure;
bar([finalComparison.CV_MeanAUC finalComparison.Test_AUC]);
xticklabels(finalComparison.Model);
ylabel("AUC");
title("AUC Comparison: CV Mean vs Test");
legend("CV Mean AUC","Test AUC","Location","best");
grid on;

if saveFigs
    exportgraphics(fComp, fullfile(figDir,"12_auc_comparison_cv_vs_test.png"), "Resolution", 300);
end


%% Feature importance (Decision Tree) 
imp = predictorImportance(tree);
[impS, idx] = sort(imp, "descend");
topK = min(20, numel(impS));

f5 = figure;
bar(impS(1:topK));
title("Decision Tree Feature Importance (Top 20) - Test Run");
xticks(1:topK);
xticklabels(featNames(idx(1:topK)));
xtickangle(45);
ylabel("Importance");

if saveFigs
    exportgraphics(f5, fullfile(figDir,"07_tree_feature_importance_top20.png"), "Resolution", 300);
end

%% Save a text summary of the decision tree model
if saveFigs
    txtFile = fullfile(figDir, "08_tree_model_summary.txt");
    fid = fopen(txtFile, "w");
    fprintf(fid, "Decision Tree Model Summary\n\n");
    fprintf(fid, "%s\n", evalc("disp(tree)"));
    fclose(fid);
end

disp("DONE.");

%% ================== Helper functions ==================
function [X, featNames] = designMatrixFromTable(T, featNamesRef)
% Creates numeric matrix from a table:
% numeric columns -> append
% categorical columns -> dummyvar
% If featNamesRef provided, aligns columns to reference (for test set).

vars = T.Properties.VariableNames;
X = [];
featNames = strings(1,0);

for i = 1:numel(vars)
    vname = vars{i};
    col = T.(vname);

    if isnumeric(col) || islogical(col)
        X = [X, double(col)];
        featNames = [featNames, string(vname)];
    else
        if ~iscategorical(col)
            col = categorical(col);
        end
        cats = categories(col);
        D = dummyvar(col);
        X = [X, D];
        featNames = [featNames, strcat(string(vname), "_", string(cats'))];
    end
end

if nargin == 2
    ref = featNamesRef;
    Xaligned = zeros(height(T), numel(ref));
    [lia, locb] = ismember(featNames, ref);
    Xaligned(:, locb(lia)) = X(:, lia);
    X = Xaligned;
    featNames = ref;
end
end

function showMetrics(Ytrue, Ypred, scores, modelName)
% Prints Accuracy, Precision, Recall, F1, AUC for positive class "Yes"
posClass = "Yes";

tp = sum((Ypred==posClass) & (Ytrue==posClass));
tn = sum((Ypred~=posClass) & (Ytrue~=posClass));
fp = sum((Ypred==posClass) & (Ytrue~=posClass));
fn = sum((Ypred~=posClass) & (Ytrue==posClass));

acc  = (tp+tn) / max(1,(tp+tn+fp+fn));
prec = tp / max(1,(tp+fp));
rec  = tp / max(1,(tp+fn));
f1   = 2*(prec*rec) / max(eps,(prec+rec));

[~,~,~,auc] = perfcurve(Ytrue, scores(:,2), posClass);

fprintf("%s | Acc=%.3f  Prec=%.3f  Rec=%.3f  F1=%.3f  AUC=%.3f\n", ...
    modelName, acc, prec, rec, f1, auc);
end



function out = computeMetricsTable(Ytrue, Ypred, scores, modelName, aucVal)
% Returns a 1-row table of key test metrics (same logic as showMetrics)

posClass = "Yes";

tp = sum((Ypred==posClass) & (Ytrue==posClass));
tn = sum((Ypred~=posClass) & (Ytrue~=posClass));
fp = sum((Ypred==posClass) & (Ytrue~=posClass));
fn = sum((Ypred~=posClass) & (Ytrue==posClass));

acc  = (tp+tn) / max(1,(tp+tn+fp+fn));
prec = tp / max(1,(tp+fp));
rec  = tp / max(1,(tp+fn));
f1   = 2*(prec*rec) / max(eps,(prec+rec));

out = table(string(modelName), acc, prec, rec, f1, aucVal, ...
    'VariableNames', {'Model','Accuracy','Precision','Recall','F1','AUC'});
end
