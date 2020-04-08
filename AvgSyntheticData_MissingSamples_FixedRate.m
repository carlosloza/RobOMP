%%
% Script that compares the performance of sparse coding variants considered
% in "RobOMP: Robust variants of Orthogonal Matching Pursuit for sparse 
% representations" DOI: 10.7717/peerj-cs.192 (open access)
% https://github.carlosloza/RobOMP
% Methodology:
% 1. A dictionary (D) with atoms from a random density (Normal) are generated
% 2. Samples that are sparsely encoded by D are generated (ground truth 
% sparsity level is provided)
% 3. Randomly selected entries are zeroed (missing entries)
% 4. The following sparse coders are implemented:
%   - Orthogonal Matching Pursuit (OMP)
%   - Generalized OMP (with optional set of number of atoms per iteration)
%   - Correntropy Matching Pursuit
%   - Robust m-estimator-based variants of OMP: Fair, Cauchy, Huber, Tukey,
%   Welsh
% 5. Performance measure: Average normalized L2-norm of difference between 
% ground truth sparse code and estimated sparse code
% Note: Sparsity level of sparse coders is set incrementally until reaching 
% ground truth
% Note: Fixed missing entries rate
% Note: Several number of iterations are allowed
% Note: Execution time is not tracked
%
% This script replicates the results summarized in Fig 3 of RobOMP article
% CORRECTION: The original results in Fig 3 for gOMP overestimated the
% sparsity level, therefore, the normalized norm was larger than the (right)
% results obtained via this script
% Also, the results in the article took a random seed so the final outputs
% of this script might not exactly match the published results.
% Lastly, this new version implements a warm start of RobOMP by default, i.e.
% the Huber solution is the initial solution for every RobOMP case.
% Empirically, this initialization was not only proved to be more stable,
% but it also yielded better performance.

close all
clearvars
clc

m = 100;                        % Dimensionality
n = 500;                        % Number of atoms
K0 = 10;                        % Ground truth sparsity level
K_v = 1:K0;                     % Set of incremental sparsity levels
nK = length(K_v);
zeroentries = 0.2;              % Rate of missing entries
N0_v = [5 10 20];               % Set of number of atoms extracted per iteration by gOMP

rng(34)                         % For reproducibility

n_it = 100;                     % Number of iterations

% Error
err_OMP = zeros(nK, n_it);
err_gOMP = zeros(nK, n_it, length(N0_v));
err_CMP = zeros(nK, n_it);
err_CauchyOMP = zeros(nK, n_it);
err_FairOMP = zeros(nK, n_it);
err_HuberOMP = zeros(nK, n_it);
err_TukeyOMP = zeros(nK, n_it);
err_WelschOMP = zeros(nK, n_it);

fprintf('Average performance of sparse coders \nSynthetic data \nRandom missing entries\n')
fprintf('Ground truth sparsity level: %u \n', K0)
fprintf('Rate of missing entries: %.2f \n', zeroentries)
fprintf('Number of iterations per case: %u \n', n_it)
fprintf('Running... \n')
for it = 1:n_it
    % Synthetic dictionary
    D = randn(m, n);
    D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));    % Normalized atoms
    x0 = zeros(n, 1);
    x0(randperm(n, K0)) = randn(K0, 1);       % Ground truth sparse code
    y = D*x0;
    % Randomly selected entries are set to zero (missing entries)
    y(randperm(m, round(zeroentries*m))) = zeros(round(zeroentries*m), 1);
    
    % OMP
    [~, ~, XOMP, ~] = gOMP(y, D, 'N0', 1, 'nnonzero', K0);
    err_OMP(:, it) = (sum((x0 - XOMP).^2,1)/norm(x0))';
    
    % GOMP   
    for i = 1:nK
        for j = 1:length(N0_v)
            xGOMP = gOMP(y, D, 'N0', N0_v(j), 'nnonzero', K_v(i));
            err_gOMP(i, it, j) = norm(x0 - xGOMP)/norm(x0);
        end
    end
    
    % Correntropy Matching Pursuit (CMP)
    [~, ~, ~, XCMP, ~] = CMP(y, D, 'nnonzero', K0);
    err_CMP(:, it) = (sum((x0 - XCMP).^2,1)/norm(x0))';
    
    % CauchyOMP
    [~, ~, ~, XCauchyOMP, ~] = RobOMP(y, D, 'm-est', 'Cauchy', 'nnonzero', K0);
    err_CauchyOMP(:, it) = (sum((x0 - XCauchyOMP).^2,1)/norm(x0))';
    
    % FairOMP
    [~, ~, ~, XFairOMP, ~] = RobOMP(y, D, 'm-est', 'Fair', 'nnonzero', K0);
    err_FairOMP(:, it) = (sum((x0 - XFairOMP).^2,1)/norm(x0))';
    
    % HuberOMP
    [~, ~, ~, XHuberOMP, ~] = RobOMP(y, D, 'm-est', 'Huber', 'nnonzero', K0);
    err_HuberOMP(:, it) = (sum((x0 - XHuberOMP).^2,1)/norm(x0))';
    
    % TukeyOMP
    [~, ~, ~, XTukeyOMP, ~] = RobOMP(y, D, 'm-est', 'Tukey', 'nnonzero', K0);
    err_TukeyOMP(:, it) = (sum((x0 - XTukeyOMP).^2,1)/norm(x0))';
    
    % WelschOMP
    [~, ~, ~, XWelschOMP, ~] = RobOMP(y, D, 'm-est', 'Welsch', 'nnonzero', K0);
    err_WelschOMP(:, it) = (sum((x0 - XWelschOMP).^2,1)/norm(x0))';
end
fprintf('Done!\n')

%% Plot results
figure('units','normalized','outerposition',[0 0 1 1])
FontSize = 40;
FontSizeLegend = 26;
Linewidth = 5;
MarkerSize = 20;
plot(K_v, mean(err_OMP,2), '--+' , 'Color', [0 0 153]/255)
hold on
% Best case for gOMP
idxgOMP = 1;
plot(K_v, mean(err_gOMP(:,:,idxgOMP),2), '--x' , 'Color', [0 102 204]/255)
plot(K_v, mean(err_CMP,2), '--d' , 'Color', [0 0 0]/255)
plot(K_v, mean(err_CauchyOMP,2), '--^' , 'Color', [76 153 0]/255)
plot(K_v, mean(err_FairOMP,2), '--v' , 'Color', [102 0 102]/255)
plot(K_v, mean(err_HuberOMP,2), '-->' , 'Color', [255 0 0]/255)
plot(K_v, mean(err_TukeyOMP,2), '--<' , 'Color', [255 128 0]/255)
plot(K_v, mean(err_WelschOMP,2), '--o' , 'Color', [128 128 128]/255)
ylabel('Norm of sparse code error')
xlabel('Matching pursuit iteration (K)')
set(findall(gcf,'-property','FontSize'),'FontSize',FontSize)
set(findall(gcf,'-property','Linewidth'),'Linewidth',Linewidth)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',MarkerSize)
legend({'OMP',['gOMP, N_0=' num2str(N0_v(idxgOMP))],'CMP','Cauchy','Fair','Huber','Tukey','Welsch'},...
    'Location','Northeast','FontSize',FontSizeLegend);
xlim([K_v(1) K_v(end)])