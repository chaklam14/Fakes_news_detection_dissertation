clear all
close all

df = readtable('gossip_SMOTE.csv');
test = readtable('gossip_SMOTE_test.csv');
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
train = xx;
test = test(:,{'class','sp_count','up_count','title_cha','title_word','cha_word','up_cha','up_word','bad_word'});

X = train(:,2:9);
Y = train(:,1);
%% label

actualy = table2array(test);
actualy = actualy(:,1);
actual1 = sum(actualy(:)==1);
actual0 = sum(actualy(:)==0);
%% Forest with 100 trees
B = TreeBagger(100,X,Y,'OOBPrediction','On');

%%
figure;
oobErrorBaggedEnsemble = oobError(B);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

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

