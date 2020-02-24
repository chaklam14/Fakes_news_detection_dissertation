clear all
close all

df = readtable('politic_combine.csv');
%%
x = df(:,{'class','sp_count','up_count','title_cha','title_word','cha_word','up_cha','up_word','bad_word'});
xx = array2table(zeros(0,9), 'VariableNames',{'class','sp_count','up_count','title_cha','title_word','cha_word','up_cha','up_word','bad_word'});

%% randomise data
s = RandStream('mt19937ar','Seed',10);
rand_pos = randperm(s,height(x)); %array of random positions

% xx is the randomised table
for k = 1:length(rand_pos)
    i = rand_pos(k);
    a = x(i,:);
    xx = [xx; a];
end
%% train and test sets
% 70% train, 30% test
n = height(xx);
train = xx(1:floor(n*0.7),:);
test = xx(ceil(n*0.7):end,:);


%% label

actualy = table2array(test);
actualy = actualy(:,1);
actual1 = sum(actualy(:)==1);
actual0 = sum(actualy(:)==0);

%%
%%%%%%%%%%%%%%%%% different depth %%%%%%%%%%%%%%%%%
trees = {};
list = linspace(0,200,41);
for i = 1:length(list)
k = list(i);
tree = fitctree(train,'class','CategoricalPredictors',2,'MaxNumSplits',k);
trees{i} = tree;
end

%% predict
labels = {};
for i = 1:length(list)
tree = trees{i};
label = predict(tree,test);
labels{i} = label;
end

con_mat = {};
for i = 1:length(list)
label = labels{i};
conmat = confusionmat(actualy,label);
con_mat{i} = conmat;
end

%%
accuracy = [];
MCC = [];

for i = 1:length(list)
a = con_mat{i};
TP = a(4);
TN = a(1);
FN = a(3);
FP = a(2);
acc = (TP+TN)/(TP+TN+FN+FP);
mcc = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));

accuracy = [accuracy acc];
MCC = [MCC mcc];
end
%%
figure()
subplot(1,2,1)
plot(list,MCC)
xlim([min(list) max(list)])
title('MCC against depth of tree')
xlabel('depth')
ylabel('MCC')
subplot(1,2,2)
plot(list,accuracy)
title('accuracy against depth of tree')
ylabel('accuracy')
xlabel('depth')
xlim([min(list) max(list)])


%% cross validation
depths = linspace(0,100,20);

rng('default')
N = numel(depths);
err = zeros(N,1);
for n=1:N
    t = fitctree(xx,'class','CrossVal','On',...
        'MaxNumSplits',depths(n));
    err(n) = kfoldLoss(t);
end
plot(depths,err);
xlabel('max. depth');
ylabel('cross-validated error');


%%
%%%%%%%%%%%%% model depth = 5 %%%%%%%%%%%%%%%%%
tree1 = fitctree(train,'class','CategoricalPredictors',2,'MaxNumSplits',5);

%% view tree
view(tree1,'mode','graph') 

imp = predictorImportance(tree1);
cvtree1 = crossval(tree1);
mse1 = kfoldLoss(cvtree1);

figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = tree1.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';


%% confusion matrix
tree1_label = predict(tree1,test);

con_mat1 = confusionmat(actualy,tree1_label);
TP = con_mat1(4);
TN = con_mat1(1);
FN = con_mat1(3);
FP = con_mat1(2);
accuracy1 = (TP+TN)/(TP+TN+FN+FP); % accuracy
MCC1 = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))); % MCC

%%
%%%%%%%% tree pruning %%%%%%%%%%%%%%
tree2 = fitctree(train,'class','Prune','on');

%% view tree
view(tree2,'mode','graph') 

imp = predictorImportance(tree2);

figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = tree1.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';


%% confusion matrix
tree2_label = predict(tree2,test);

con_mat2 = confusionmat(actualy,tree2_label);
TP = con_mat2(4);
TN = con_mat2(1);
FN = con_mat2(3);
FP = con_mat2(2);
accuracy2 = (TP+TN)/(TP+TN+FN+FP); % accuracy
MCC2 = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))); % MCC

%%
%% unconstrained depth
tree3 = fitctree(train,'class','CategoricalPredictors',2);

%% view tree
view(tree3,'mode','graph') 

imp = predictorImportance(tree3);

figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = tree1.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

%%
%%
tree3_label = predict(tree3,test);
%%
con_mat3 = confusionmat(actualy,tree3_label);

TP = con_mat3(4);
TN = con_mat3(1);
FN = con_mat3(3);
FP = con_mat3(2);
accuracy3 = (TP+TN)/(TP+TN+FN+FP);
MCC3 = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));