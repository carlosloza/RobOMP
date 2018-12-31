%%
% Comparison of Robust OMP methods with OMP and GOMP
% Student's t noise
% Synthetic Data

close all
clearvars
clc

m = 100;        % Dimensionality
n = 500;        % Number of atoms
K = 10;
degf_v = 1:10;       
ndeg = length(degf_v);
N0_v = 2:5;

n_it = 100;

% Error
err_OMP = zeros(ndeg, n_it);
err_GOMP = zeros(ndeg, n_it, length(N0_v));
err_CorrOMP = zeros(ndeg, n_it);
err_CauchyOMP = zeros(ndeg, n_it);
err_FairOMP = zeros(ndeg, n_it);
err_HuberOMP = zeros(ndeg, n_it);
err_TukeyOMP = zeros(ndeg, n_it);
err_WelschOMP = zeros(ndeg, n_it);

% Time
time_OMP = zeros(ndeg, n_it);
time_GOMP = zeros(ndeg, n_it, length(N0_v));
time_CorrOMP = zeros(ndeg, n_it);
time_CauchyOMP = zeros(ndeg, n_it);
time_FairOMP = zeros(ndeg, n_it);
time_HuberOMP = zeros(ndeg, n_it);
time_TukeyOMP = zeros(ndeg, n_it);
time_WelschOMP = zeros(ndeg, n_it);

for i = 1:ndeg
    for it = 1:n_it
        display(['Degrees of Freedom ' num2str(degf_v(i)) ', Iteration ' num2str(it)])
        D = randn(m, n);
        D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));
        x0 = zeros(n, 1);
        x0(randperm(n, K)) = randn(K, 1);
        y = D*x0;
        
        y = y + trnd(degf_v(i), size(y));
        
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
        xCorrOMP = CorrOMP(y, D, K);
        time_CorrOMP(i, it) = toc;
        err_CorrOMP(i, it) = norm(x0 - xCorrOMP(:,end))/norm(x0);
        
        % CauchyOMP
        disp('CauchyOMP')
        tic
        xCauchyOMP = CauchyOMP(y, D, K);
        time_CauchyOMP(i, it) = toc;
        err_CauchyOMP(i, it) = norm(x0 - xCauchyOMP(:,end))/norm(x0);
        
        % FairOMP
        disp('FairOMP')
        tic
        xFairOMP = FairOMP(y, D, K);
        time_FairOMP(i, it) = toc;
        err_FairOMP(i, it) = norm(x0 - xFairOMP(:,end))/norm(x0);
        
        % HuberOMP
        disp('HuberOMP')
        tic
        xHuberOMP = HuberOMP(y, D, K);
        time_HuberOMP(i, it) = toc;
        err_HuberOMP(i, it) = norm(x0 - xHuberOMP(:,end))/norm(x0);
        
        % TukeyOMP
        disp('TukeyOMP')
        tic
        xTukeyOMP = TukeyOMP(y, D, K);
        time_TukeyOMP(i, it) = toc;
        err_TukeyOMP(i, it) = norm(x0 - xTukeyOMP(:,end))/norm(x0);
        
        % WelschOMP
        disp('WelschOMP')
        tic
        xWelschOMP = WelschOMP(y, D, K);
        time_WelschOMP(i, it) = toc;
        err_WelschOMP(i, it) = norm(x0 - xWelschOMP(:,end))/norm(x0);
    end
end
