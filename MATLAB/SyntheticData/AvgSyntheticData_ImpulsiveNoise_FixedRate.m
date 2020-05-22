%%
% Script that compares the performance of sparse coding variants considered
% in "RobOMP: Robust variants of Orthogonal Matching Pursuit for sparse 
% representations" DOI: 10.7717/peerj-cs.192 (open access)
% Fixed rate impulsive noise scenario
% Author: Carlos Loza
% https://github.carlosloza/RobOMP
%
% Methodology:
% ------------
% 1. A dictionary (D) with atoms from a random density (Normal) are generated
% 2. Samples that are sparsely encoded by D are generated (ground truth 
% sparsity level is provided)
% 3. Randomly selected entries are affected by additive noise, i.e. impulsive
% noise, with a given SNR
% 4. The following sparse coders are implemented:
%   - Orthogonal Matching Pursuit (OMP)
%   - Generalized OMP (with optional set of number of atoms per run)
%   - Correntropy Matching Pursuit
%   - Robust m-estimator-based variants of OMP: Fair, Cauchy, Huber, Tukey,
%   Welsh
% 5. Performance measure: Average normalized L2-norm of difference between 
% ground truth sparse code and estimated sparse code
% 
% Notes:
% ------
% - Sparsity level of sparse coders is set equal to ground truth
% - Fixed affected entries rate
% - Several additive impulsive noise SNR and number of iterations are allowed
% - Execution time is tracked as well
% - This script replicates the results summarized in Fig 4B of RobOMP article
% - The results in the article took a random seed so the final outputs of
% this script might not exactly match the published results
% - Lastly, this new version implements a warm start of RobOMP by default, 
% i.e. the Huber solution is the initial solution for every RobOMP case
% Empirically, this initialization was not only proved to be more stable,
% but it also yielded better performance
%
% CORRECTION: The original results in Fig 4B for gOMP overestimated the
% sparsity level, therefore, the normalized norm was larger than the (right)
% results obtained via this script 

close all
clearvars
clc

addpath('..')               % Assuming directories as in remote repo

m = 100;                        % Dimensionality
n = 500;                        % Number of atoms
K = 10;                         % Ground truth sparsity level
N0_v = [5 10 20];               % Set of number of atoms extracted per iteration by gOMP
SNRdB_v = -25:5:0;              % Set of additive impulsive noise SNR
nSNR = length(SNRdB_v);

impulentries = 0.1;             % Rate of affected entries

rng(34)                         % For reproducibility

n_it = 100;                     % Number of iterations

% Error
err_OMP = zeros(nSNR, n_it);
err_gOMP = zeros(nSNR, n_it, length(N0_v));
err_CMP = zeros(nSNR, n_it);
err_CauchyOMP = zeros(nSNR, n_it);
err_FairOMP = zeros(nSNR, n_it);
err_HuberOMP = zeros(nSNR, n_it);
err_TukeyOMP = zeros(nSNR, n_it);
err_WelschOMP = zeros(nSNR, n_it);

% Time
time_OMP = zeros(nSNR, n_it);
time_gOMP = zeros(nSNR, n_it, length(N0_v));
time_CMP = zeros(nSNR, n_it);
time_CauchyOMP = zeros(nSNR, n_it);
time_FairOMP = zeros(nSNR, n_it);
time_HuberOMP = zeros(nSNR, n_it);
time_TukeyOMP = zeros(nSNR, n_it);
time_WelschOMP = zeros(nSNR, n_it);

fprintf('Average performance of sparse coders \nSynthetic data \n')
fprintf('Random entries affected by additive noise, i.e. impulsive noise\n')
fprintf('Ground truth sparsity level: %u \n', K)
fprintf('Rate of entries affected by noise: %.2f \n', impulentries)
fprintf('Number of iterations per case: %u \n', n_it)
for i = 1:nSNR
    fprintf('Additive impulsive noise SNR: %.2f dB \n', SNRdB_v(i))
    for it = 1:n_it
        % Synthetic dictionary
        D = randn(m, n);
        D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));    % Normalized atoms
        x0 = zeros(n, 1);
        x0(randperm(n, K)) = randn(K, 1);       % Ground truth sparse code
        y = D*x0;
        % Randomly selected entries affected by additive impulsive noise
        yimp = AWGNoise(y, SNRdB_v(i));
        imp_idx = randperm(m, round(impulentries*m));
        y(imp_idx) = yimp(imp_idx);
        
        % OMP
        tic
        xOMP = gOMP(y, D, 'N0', 1, 'nnonzero', K);
        time_OMP(i, it) = toc;
        err_OMP(i, it) = norm(x0 - xOMP)/norm(x0);
        
        % GOMP
        for j = 1:length(N0_v)
            tic
            xGOMP = gOMP(y, D, 'N0', N0_v(j), 'nnonzero', K);
            time_gOMP(i, it, j) = toc;
            err_gOMP(i, it, j) = norm(x0 - xGOMP)/norm(x0);
        end
        
        % Correntropy Matching Pursuit (CMP)
        tic
        xCMP = CMP(y, D, 'nnonzero', K);
        time_CMP(i, it) = toc;
        err_CMP(i, it) = norm(x0 - xCMP)/norm(x0);
        
        % CauchyOMP
        tic
        xCauchyOMP = RobOMP(y, D, 'm-est', 'Cauchy', 'nnonzero', K);
        time_CauchyOMP(i, it) = toc;
        err_CauchyOMP(i, it) = norm(x0 - xCauchyOMP)/norm(x0);
        
        % FairOMP
        tic
        xFairOMP = RobOMP(y, D, 'm-est', 'Fair', 'nnonzero', K);
        time_FairOMP(i, it) = toc;
        err_FairOMP(i, it) = norm(x0 - xFairOMP)/norm(x0);
        
        % HuberOMP
        tic
        xHuberOMP = RobOMP(y, D, 'm-est', 'Huber', 'nnonzero', K);
        time_HuberOMP(i, it) = toc;
        err_HuberOMP(i, it) = norm(x0 - xHuberOMP)/norm(x0);
        
        % TukeyOMP
        tic
        xTukeyOMP = RobOMP(y, D, 'm-est', 'Tukey', 'nnonzero', K);
        time_TukeyOMP(i, it) = toc;
        err_TukeyOMP(i, it) = norm(x0 - xTukeyOMP)/norm(x0);
        
        % WelschOMP
        tic
        xWelschOMP = RobOMP(y, D, 'm-est', 'Welsch', 'nnonzero', K);
        time_WelschOMP(i, it) = toc;
        err_WelschOMP(i, it) = norm(x0 - xWelschOMP)/norm(x0);
    end
end

%% Plot results - Fig. 4B - Logarithm for visual purposes
figure('units','normalized','outerposition',[0 0 1 1])
FontSize = 40;
FontSizeLegend = 23;
Linewidth = 5;
MarkerSize = 20;
plot(SNRdB_v, log10(mean(err_OMP,2)), '--+' , 'Color', [0 0 153]/255)
hold on
% Best case for gOMP
idxgOMP = 2;
plot(SNRdB_v, log10(mean(err_gOMP(:,:,idxgOMP),2)), '--x' , 'Color', [0 102 204]/255)
plot(SNRdB_v, log10(mean(err_CMP,2)), '--d' , 'Color', [0 0 0]/255)
plot(SNRdB_v, log10(mean(err_CauchyOMP,2)), '--^' , 'Color', [76 153 0]/255)
plot(SNRdB_v, log10(mean(err_FairOMP,2)), '--v' , 'Color', [102 0 102]/255)
plot(SNRdB_v, log10(mean(err_HuberOMP,2)), '-->' , 'Color', [255 0 0]/255)
plot(SNRdB_v, log10(mean(err_TukeyOMP,2)), '--<' , 'Color', [255 128 0]/255)
plot(SNRdB_v, log10(mean(err_WelschOMP,2)), '--o' , 'Color', [128 128 128]/255)
ylabel('Sparse code error norm (log)')
xlabel(['Additive impulsive noise SNR (dB). Rate = ' num2str(impulentries)])
set(findall(gcf,'-property','FontSize'),'FontSize',FontSize)
set(findall(gcf,'-property','Linewidth'),'Linewidth',Linewidth)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',MarkerSize)
legend({'OMP',['gOMP, N_0=' num2str(N0_v(idxgOMP))],'CMP','Cauchy','Fair','Huber','Tukey','Welsch'},...
    'Location','Southwest','FontSize',FontSizeLegend);
ylim([-6 1])

%% Plot times in miliseconds
figure('units','normalized','outerposition',[0 0 1 1])
FontSize = 40;
FontSizeLegend = 22;
Linewidth = 5;
MarkerSize = 20;
plot(SNRdB_v, 1000*mean(time_OMP,2), '--+' , 'Color', [0 0 153]/255)
hold on
plot(SNRdB_v, 1000*mean(time_gOMP(:,:,idxgOMP),2), '--x' , 'Color', [0 102 204]/255)
plot(SNRdB_v, 1000*mean(time_CMP,2), '--d' , 'Color', [0 0 0]/255)
plot(SNRdB_v, 1000*mean(time_CauchyOMP,2), '--^' , 'Color', [76 153 0]/255)
plot(SNRdB_v, 1000*mean(time_FairOMP,2), '--v' , 'Color', [102 0 102]/255)
plot(SNRdB_v, 1000*mean(time_HuberOMP,2), '-->' , 'Color', [255 0 0]/255)
plot(SNRdB_v, 1000*mean(time_TukeyOMP,2), '--<' , 'Color', [255 128 0]/255)
plot(SNRdB_v, 1000*mean(time_WelschOMP,2), '--o' , 'Color', [128 128 128]/255)
ylabel('Processing time (ms.)')
xlabel(['Additive impulsive noise SNR (dB). Rate = ' num2str(impulentries)])
set(findall(gcf,'-property','FontSize'),'FontSize',FontSize)
set(findall(gcf,'-property','Linewidth'),'Linewidth',Linewidth)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',MarkerSize)
legend({'OMP',['gOMP, N_0=' num2str(N0_v(idxgOMP))],'CMP','Cauchy','Fair','Huber','Tukey','Welsch'},...
    'Location','Northwest','FontSize',FontSizeLegend);
ylim([0 20])