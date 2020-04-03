%%
% Comparison of Robust OMP methods with OMP and GOMP
% Gaussian noise
% Synthetic Data

close all
clearvars
clc

m = 100;        % Dimensionality
n = 500;        % Number of atoms
K = 10;
sigma_v = 0:0.5:5;        
nSigma = length(sigma_v);
N0_v = 2:5;

rng(34)         % For reproducibility

n_it = 10;

% Error
err_OMP = zeros(nSigma, n_it);
err_GOMP = zeros(nSigma, n_it, length(N0_v));
err_CorrOMP = zeros(nSigma, n_it);
err_CauchyOMP = zeros(nSigma, n_it);
err_FairOMP = zeros(nSigma, n_it);
err_HuberOMP = zeros(nSigma, n_it);
err_TukeyOMP = zeros(nSigma, n_it);
err_WelschOMP = zeros(nSigma, n_it);

% Time
time_OMP = zeros(nSigma, n_it);
time_GOMP = zeros(nSigma, n_it, length(N0_v));
time_CorrOMP = zeros(nSigma, n_it);
time_CauchyOMP = zeros(nSigma, n_it);
time_FairOMP = zeros(nSigma, n_it);
time_HuberOMP = zeros(nSigma, n_it);
time_TukeyOMP = zeros(nSigma, n_it);
time_WelschOMP = zeros(nSigma, n_it);

for i = 1:nSigma
    for it = 1:n_it
        display(['SNR ' num2str(sigma_v(i)) ', Iteration ' num2str(it)])
        D = randn(m, n);
        D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));
        x0 = zeros(n, 1);
        x0(randperm(n, K)) = randn(K, 1);
        y = D*x0;
        
        %ynoi = add_AWGN(y, sigma_v(i));
        ynoi = y + sigma_v(i)*randn(size(y));
        y = ynoi;
        
        % OMP
        disp('OMP')
        tic
        [yOMP, res, coeff, idx] = wmpalg('OMP', y, D, 'itermax', K);
        time_OMP(i, it) = toc;
        xOMP = zeros(n, 1);
        xOMP(idx) = coeff;
        err_OMP(i, it) = norm(x0 - xOMP)/norm(x0);
        
        % GOMP
        disp('GOMP')
        for j = 1:length(N0_v)
            tic
            xGOMP = gomp(y, D, K, N0_v(j));
            time_GOMP(i, it, j) = toc;
            err_GOMP(i, it, j) = norm(x0 - xGOMP(:,end))/norm(x0);
        end
        
        % CorrOMP
        disp('CorrOMP')
        tic
        xCorrOMP = CMP(y, D, 'K', K);
        time_CorrOMP(i, it) = toc;
        err_CorrOMP(i, it) = norm(x0 - xCorrOMP)/norm(x0);
        
        % CauchyOMP
        disp('CauchyOMP')
        tic
        xCauchyOMP = RobOMP(y, D, 'm-est', 'Cauchy', 'maxiter', K);
        time_CauchyOMP(i, it) = toc;
        err_CauchyOMP(i, it) = norm(x0 - xCauchyOMP)/norm(x0);
        
        % FairOMP
        disp('FairOMP')
        tic
        xFairOMP = RobOMP(y, D, 'm-est', 'Fair', 'maxiter', K);
        time_FairOMP(i, it) = toc;
        err_FairOMP(i, it) = norm(x0 - xFairOMP)/norm(x0);
        
        % HuberOMP
        disp('HuberOMP')
        tic
        xHuberOMP = RobOMP(y, D, 'm-est', 'Huber', 'maxiter', K);
        time_HuberOMP(i, it) = toc;
        err_HuberOMP(i, it) = norm(x0 - xHuberOMP)/norm(x0);
        
        % TukeyOMP
        disp('TukeyOMP')
        tic
        xTukeyOMP = RobOMP(y, D, 'm-est', 'Tukey', 'maxiter', K);
        time_TukeyOMP(i, it) = toc;
        err_TukeyOMP(i, it) = norm(x0 - xTukeyOMP)/norm(x0);
        
        % WelschOMP
        disp('WelschOMP')
        tic
        xWelschOMP = RobOMP(y, D, 'm-est', 'Welsch', 'maxiter', K);
        time_WelschOMP(i, it) = toc;
        err_WelschOMP(i, it) = norm(x0 - xWelschOMP)/norm(x0);
    end
end
