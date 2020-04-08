%%
% Script that compares the performance of sparse coding variants considered
% in "RobOMP: Robust variants of Orthogonal Matching Pursuit for sparse 
% representations" DOI: 10.7717/peerj-cs.192 (open access)
% https://github.carlosloza/RobOMP
% Methodology:
% 1. A dictionary (D) with atoms from a random density (Normal) are generated
% 2. Samples that are sparsely encoded by D are generated (ground truth 
% sparsity level is provided)
% 3. Additive zero-mean gaussian noise is added to samples
% 4. The following sparse coders are implemented:
%   - Orthogonal Matching Pursuit (OMP)
%   - Generalized OMP (with optional set of number of atoms per iteration)
%   - Correntropy Matching Pursuit
%   - Robust m-estimator-based variants of OMP: Fair, Cauchy, Huber, Tukey,
%   Welsh
% 5. Performance measure: Average normalized L2-norm of difference between 
% ground truth sparse code and estimated sparse code
% Note: Several noise standard deviations and number of iterations are allowed
% Note: Execution time is tracked as well
%
% Setting the standard deviation to 2 will yield the averages summarized in
% Table 3 of RobOMP article
% CORRECTION: The original results in Table 3 for gOMP overestimated the
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

m = 100;                    % Dimensionality
n = 500;                    % Number of atoms
K = 10;                     % Ground truth sparsity level
sigma_v = 0.5:0.5:5;        % Set of standard deviations of added zero-mean gaussian noise       
nSigma = length(sigma_v);
N0_v = [5 10 20];           % Set of number of atoms extracted per iteration by gOMP

rng(34)                     % For reproducibility

n_it = 100;                 % Number of iterations

% Error
err_OMP = zeros(nSigma, n_it);
err_gOMP = zeros(nSigma, n_it, length(N0_v));
err_CMP = zeros(nSigma, n_it);
err_CauchyOMP = zeros(nSigma, n_it);
err_FairOMP = zeros(nSigma, n_it);
err_HuberOMP = zeros(nSigma, n_it);
err_TukeyOMP = zeros(nSigma, n_it);
err_WelschOMP = zeros(nSigma, n_it);

% Time
time_OMP = zeros(nSigma, n_it);
time_gOMP = zeros(nSigma, n_it, length(N0_v));
time_CMP = zeros(nSigma, n_it);
time_CauchyOMP = zeros(nSigma, n_it);
time_FairOMP = zeros(nSigma, n_it);
time_HuberOMP = zeros(nSigma, n_it);
time_TukeyOMP = zeros(nSigma, n_it);
time_WelschOMP = zeros(nSigma, n_it);

fprintf('Average performance of sparse coders \nSynthetic data \nAdditive Gaussian noise\n')
fprintf('Ground truth sparsity level: %u \n', K)
fprintf('Number of iterations per case: %u \n', n_it)
for i = 1:nSigma
    fprintf('Gaussian additive noise standard deviation: %.2f \n', sigma_v(i))
    for it = 1:n_it
        % Synthetic dictionary
        D = randn(m, n);
        D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));    % Normalized atoms
        x0 = zeros(n, 1);
        x0(randperm(n, K)) = randn(K, 1);   % Ground truth sparse code
        y = D*x0;       
        % Add zero-mean gaussian noise 
        ynoi = y + sigma_v(i)*randn(size(y));
        y = ynoi;
        
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

%% Plot results
figure('units','normalized','outerposition',[0 0 1 1])
FontSize = 40;
FontSizeLegend = 27;
Linewidth = 5;
MarkerSize = 20;
plot(sigma_v, mean(err_OMP,2), '--+' , 'Color', [0 0 153]/255)
hold on
% Best case for gOMP
idxgOMP = 3;
plot(sigma_v, mean(err_gOMP(:,:,idxgOMP),2), '--x' , 'Color', [0 102 204]/255)
plot(sigma_v, mean(err_CMP,2), '--d' , 'Color', [0 0 0]/255)
plot(sigma_v, mean(err_CauchyOMP,2), '--^' , 'Color', [76 153 0]/255)
plot(sigma_v, mean(err_FairOMP,2), '--v' , 'Color', [102 0 102]/255)
plot(sigma_v, mean(err_HuberOMP,2), '-->' , 'Color', [255 0 0]/255)
plot(sigma_v, mean(err_TukeyOMP,2), '--<' , 'Color', [255 128 0]/255)
plot(sigma_v, mean(err_WelschOMP,2), '--o' , 'Color', [128 128 128]/255)
ylabel('Norm of sparse code error')
xlabel('Gaussian additive noise standard deviation')
set(findall(gcf,'-property','FontSize'),'FontSize',FontSize)
set(findall(gcf,'-property','Linewidth'),'Linewidth',Linewidth)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',MarkerSize)
legend({'OMP',['gOMP, N_0=' num2str(N0_v(idxgOMP))],'CMP','Cauchy','Fair','Huber','Tukey','Welsch'},...
    'Location','Northwest','FontSize',FontSizeLegend);
xlim([sigma_v(1) sigma_v(end)])

%% Plot times in miliseconds
figure('units','normalized','outerposition',[0 0 1 1])
FontSize = 40;
FontSizeLegend = 23;
Linewidth = 5;
MarkerSize = 20;
plot(sigma_v, 1000*mean(time_OMP,2), '--+' , 'Color', [0 0 153]/255)
hold on
plot(sigma_v, 1000*mean(time_gOMP(:,:,idxgOMP),2), '--x' , 'Color', [0 102 204]/255)
plot(sigma_v, 1000*mean(time_CMP,2), '--d' , 'Color', [0 0 0]/255)
plot(sigma_v, 1000*mean(time_CauchyOMP,2), '--^' , 'Color', [76 153 0]/255)
plot(sigma_v, 1000*mean(time_FairOMP,2), '--v' , 'Color', [102 0 102]/255)
plot(sigma_v, 1000*mean(time_HuberOMP,2), '-->' , 'Color', [255 0 0]/255)
plot(sigma_v, 1000*mean(time_TukeyOMP,2), '--<' , 'Color', [255 128 0]/255)
plot(sigma_v, 1000*mean(time_WelschOMP,2), '--o' , 'Color', [128 128 128]/255)
ylabel('Processing time (ms.)')
xlabel('Gaussian additive noise standard deviation')
set(findall(gcf,'-property','FontSize'),'FontSize',FontSize)
set(findall(gcf,'-property','Linewidth'),'Linewidth',Linewidth)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',MarkerSize)
legend({'OMP',['gOMP, N_0=' num2str(N0_v(idxgOMP))],'CMP','Cauchy','Fair','Huber','Tukey','Welsch'},...
    'Location','Northwest','FontSize',FontSizeLegend);
xlim([sigma_v(1) sigma_v(end)])
ylim([0 19])
