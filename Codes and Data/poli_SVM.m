close all
clear all

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
n = height(xx);
train = xx(1:floor(n*0.7),:);
test = xx(ceil(n*0.7):end,:);

X = train(:,2:9);
Y = train(:,1);
%% train the model
SVM = fitcsvm(X,Y,'KernelFunction','linear','Standardize',true);

%% test the model
[label,score] = predict(SVM,test);
ScoreSVMModel = fitPosterior(SVM,X,Y);

test_Y = table2array(test(:,1));
con_mat = confusionmat(test_Y,label);

TP = con_mat(4);
TN = con_mat(1);
FN = con_mat(3);
FP = con_mat(2);
accuracy = (TP+TN)/n;
MCC = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));

%%
CVS1 = crossval(SVM,'KFold',5);

kfoldLoss(CVS1)

%% using gaussian
% train the model
SVM2 = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true);

% test the model
[label2,score2] = predict(SVM2,test);

con_mat2 = confusionmat(test_Y,label2);

TP = con_mat2(4);
TN = con_mat2(1);
FN = con_mat2(3);
FP = con_mat2(2);
accuracy2 = (TP+TN)/n;
MCC2 = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));


%%
CVS2 = crossval(SVM2,'KFold',5);

kfoldLoss(CVS2)

%%
% Gaussian seems to be better, so continue with soft-margin

% train the model

margin = linspace(0,100,20);
MCC3 = [];
for i = margin
    SVM3 = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true,'BoxConstraint',i);
    
    [label3,score3] = predict(SVM3,test);

    con_mat3 = confusionmat(test_Y,label3);
    TP = con_mat3(4);
    TN = con_mat3(1);
    FN = con_mat3(3);
    FP = con_mat3(2);
   
    MCC3 = [MCC3 (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))];
end
%%
[M,k] = max(MCC3);
opt = margin(k);

%%
SVM4 = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true,'BoxConstraint',opt);

[label4,score4] = predict(SVM4,test);
con_mat4 = confusionmat(test_Y,label4);

TP = con_mat4(4);
TN = con_mat4(1);
FN = con_mat4(3);
FP = con_mat4(2);
accuracy4 = (TP+TN)/n;
MCC4 = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));

%%
CVS4 = crossval(SVM4,'KFold',5);

kfoldLoss(CVS4)

