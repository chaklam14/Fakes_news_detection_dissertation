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
%% Random forest
B = TreeBagger(60,X,Y,'OOBPrediction','On','OOBPredictorImportance','On');

%%
figure;
oobErrorBaggedEnsemble = oobError(B);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';


%%

actualy = table2array(test);
actualy = actualy(:,1);
actual1 = sum(actualy(:)==1);
actual0 = sum(actualy(:)==0);


%%
B1_label = predict(B,test);
testy = str2double(B1_label);

cm1 = confusionchart(actualy,testy);
con_mat1 = confusionmat(actualy,testy);

TP = con_mat1(4);
TN = con_mat1(1);
FN = con_mat1(3);
FP = con_mat1(2);
accuracy1 = (TP+TN)/(TP+TN+FN+FP);
MCC1 = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));


%%
imp = B.OOBPermutedPredictorDeltaError;

figure;
bar(imp);
title('Standard CART');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = B.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';


%% max depth of 50
B2 = TreeBagger(43,X,Y,'OOBPrediction','On','MaxNumSplits',50);
%% label
B2_label = predict(B2,test);
testy = str2double(B2_label);

cm2 = confusionchart(actualy,testy);
con_mat2 = confusionmat(actualy,testy);

TP = con_mat2(4);
TN = con_mat2(1);
FN = con_mat2(3);
FP = con_mat2(2);
accuracy2 = (TP+TN)/(TP+TN+FN+FP);
MCC2 = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));

%% view tree

view(B2.Trees{10})

%%
figure;
oobErrorBaggedEnsemble = oobError(B2);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%% find optimal depth
depths = linspace(5,100,20);
MCC3 = [];
for i = depths
    B3 = TreeBagger(47,X,Y,'MaxNumSplits',i);
    
    [label3,score3] = predict(B3,test);
    label3 = str2double(label3);
    con_mat3 =confusionmat(actualy,label3);
    
    TP = con_mat3(4);
    TN = con_mat3(1);
    FN = con_mat3(3);
    FP = con_mat3(2);
    MCC3 = [MCC3 (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))];
end
%%
[M,k] = max(MCC3);
opt = depths(3);

% depth=15 is the best

%%
%%%% max depth of 15
B4 = TreeBagger(43,X,Y,'OOBPrediction','On','MaxNumSplits',15);
%% label
B4_label = predict(B4,test);
testy = str2double(B4_label);

cm4 = confusionchart(actualy,testy);
con_mat4 = confusionmat(actualy,testy);

TP = con_mat4(4);
TN = con_mat4(1);
FN = con_mat4(3);
FP = con_mat4(2);
accuracy4 = (TP+TN)/(TP+TN+FN+FP);
MCC4 = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));

%%
figure;
oobErrorBaggedEnsemble = oobError(B4);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
