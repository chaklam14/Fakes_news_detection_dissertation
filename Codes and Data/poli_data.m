%%%%%%%%%%%%%%%%%%% use this for gossip %%%%%%%%%%%%%%%%%%%

close all
clear all

preal = readtable('politic_real_clean.csv');
pfake = readtable('politic_fake_clean.csv');

%% REAL
x = preal(:,{'sp_count','up_count','title_cha','title_word','cha_word','up_cha','up_word'}); % note: no bad word
real = summary(x);


%% FAKE
xf = pfake(:,{'sp_count','up_count','title_cha','title_word','cha_word','up_cha','up_word'}); % note: no bad word
fake = summary(xf);


%%
variables = {'number of sepcial characters', 'number of capital letters', ...
    'number of characters', 'number of words', 'average number of characters in each word',...
    'percentage of capital letter in terms of character', 'percentage of capital letter in terms of word',};
%% Plots
% same axis, two plots
close all
for i = 1:7
    figure(i)
    
    x1 = table2array(x(:,i));
    x2 = table2array(xf(:,i));
    
    if max(x1)>max(x2)
        lim = [min(x1) max(x1)];
    else
        lim = [min(x2) max(x2)];
    end
    
    subplot(1,2,1)
    histogram(x1,'Normalization','probability')
    set(gca,'FontSize',20)
    ylabel('percentage')
    title('real news')
    ylim([0 1])
    xlim(lim)
    
    subplot(1,2,2)
    histogram(x2,'Normalization','probability','FaceColor','red')
    set(gca,'FontSize',20)
    ylabel('percentage')
    xlabel(variables(i))
    title('fake news')
    ylim([0 1])
    xlim(lim)
    
    %sgtitle(['x-axis =' variables(i)])
end

%% Plots
% same plots

for k = 1:7
    figure(k)
        
    x1 = table2array(x(:,k));
    x2 = table2array(xf(:,k));
    
    binwidth = max(x1)/10;
    
    histogram(x1,'BinWidth',binwidth,'Normalization','probability');
    hold on
    histogram(x2,'BinWidth',binwidth,'Normalization','probability','FaceColor','red');
    hold off
    set(gca,'FontSize',20)
    ylabel('percentage')
    xlabel(variables(k))
    ylim([0 1])
    title('real vs fake news')
    
end

