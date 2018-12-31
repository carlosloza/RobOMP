function [X_gomp, E_gomp] = gomp(y, A, K, S, err)

% Generalized Orthogonal Matching Pursuit (gOMP) is a greedy algorothm that
% provides approximate solution to the problem: min ||x||_0 such that Ax = y. 
% gOMP extends the conventional Orthogonal Matching Pursuit (OMP) algorithm
% by allowing S indices to be chosen per iteration.   

% Input:
% y		    : measurements
% A       	: measurement matrix
% K	    	: Sparsity level of underlying signal to be recovered
% S	    	: number of indices chosen per iteration
% err       : residual tolerant


% Output:
% X_gomp    : estimated sparse signal
% E_gomp    : estimated error

% Based on the code by Jian Wang, Seokbeop Kwon and Byonghyo Shim, 2012

residual  = y;
supp	  = [];
i = 0;

if nargin == 4
    X_gomp = zeros(size(A,2), K);
    E_gomp = zeros(length(y), K);
    while i < K
        i = i + 1;
        [~, idx] = sort(abs(A' * residual), 'descend');
        supp_temp = union(supp, idx(1:S));
        
        if isequal(supp_temp, supp)
            X_gomp(:,i:end) = repmat(X_gomp(:,i-1), 1, K - i + 1);
            E_gomp(:,i:end) = repmat(E_gomp(:,i-1), 1, K - i + 1);
            i = K;
        break
        else
            supp = supp_temp;
            x_hat = A(:,supp)\y;
            X_gomp(supp, i) = x_hat;
            residual = y - A(:,supp) * x_hat;
            E_gomp(:, i) = residual;
        end
    end
    
elseif nargin == 5
    max_it = 500;
    X_gomp = zeros(size(A,2), max_it);
    E_gomp = zeros(length(y), max_it);
    while (norm(residual) > err && i < max_it)
        i = i + 1;
        [~, idx] = sort(abs(A' * residual), 'descend');
        supp_temp = union(supp, idx(1:S));
        
        if isequal(supp_temp, supp)
            X_gomp(:,i:end) = repmat(X_gomp(:,i-1), 1, max_it - i + 1);
            E_gomp(:,i:end) = repmat(E_gomp(:,i-1), 1, max_it - i + 1);
            i = max_it;
        break
        else
            supp = supp_temp;
            x_hat = A(:,supp)\y;
            X_gomp(supp, i) = x_hat;
            residual = y - A(:,supp) * x_hat;
            E_gomp(:, i) = residual;
        end
    end
    
end

X_gomp = X_gomp(:,1:i);
E_gomp = E_gomp(:,1:i);
end