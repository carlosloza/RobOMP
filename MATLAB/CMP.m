function [x, e, w, X, E] = CMP(y, D, varargin)
% Implementation of Correntropy Matching Pursuit as introduced in Wang et
% al. 2017 (DOI: 10.1109/TCYB.2016.2544852)
% Author: Carlos Loza
% Part of RobOMP package. DOI: 10.7717/peerj-cs.192 (open access)
% https://github.carlosloza/RobOMP
%
% Parameters
% ----------
% y :       vector, size (m, 1)
%           Signal to be sparsely encoded
% D :       matrix, size (m , n)
%           Dictionary/measurement matrix made up of atoms
% nnonzero :int
%           Number of non-zero coefficients in sparse code
%           This is equal to number of iterations in Orthogonal Matching
%           Pursuit (OMP)
% tol :     float
%           Residual norm tolerance. Dispersion/power rate not explained by
%           the sparse code with respect to the norm of y
%           Default: 0.1 i.e. 10% of the L2 norm of input y
%
% If neither K nor tol are set, then tol is set to default
% If both K and tol are set, then the algorithm stops when both conditions
% are met
%
% Returns
% -------
% x :       vector, size (n, 1)
%           Sparse code corresponding to y encoded by D with sparsity level K
% e :       vector, size (m, 1)
%           Residue/error after sparse coding of y with sparsity level K
% w :       vector, size (m, 1)
%           Weights associated to entries of y according to correntropy-based
%           linear regression
% X :       matrix, size (n, K)
%           Same as x, but each column corresponds to decreasingly sparser
%           solutions, i.e. first column has sparsirty level of 1, second
%           column has sparsity level of 2, and so on
% E :       matrix, size (m, K)
%           Same as e, but each column corresponds to residue after  
%           decreasingly sparser solutions, i.e. likewise X
%
% Example: x = CMP(y, D, 'nnonzero', 10, 'tol', 0.25)

[m, n] = size(D);
% Check inputs
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'nnonzero')
        nnonzero = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'tol')
        tol = varargin{i + 1};
    end    
end

% Set defaults if variables were not set
if exist('nnonzero', 'var') && ~exist('tol', 'var')
    flcase = 1;         % Stopping criterion based solely on K
elseif ~exist('nnonzero', 'var') && exist('tol', 'var') 
    flcase = 2;         % Stopping criterion based solely on tol
elseif ~exist('nnonzero', 'var') && ~exist('tol', 'var')
    flcase = 3;         % Stopping criterion based on both tol alone
end

switch flcase
    case 1
        tol = -1;
    case 2
        nnonzero = n;           % Extreme case
    case 3
        nnonzero = n;           % Extreme case
        tol = 0.1;
end

normy = norm(y);
r = y;
i = 0;
X = zeros(n, nnonzero);
E = zeros(m, nnonzero);
idx_spcode = [];
while (norm(r)/normy > tol && i < nnonzero)
    i = i + 1;
    abscorr = abs(r'*D);
    [~, idx] = max(abscorr);
    if any(diff(sort([idx_spcode idx])) == 0)       % Case for repeated atom
        X(:,i:end) = repmat(X(:,i-1), 1, nnonzero - i + 1);
        E(:,i:end) = repmat(E(:,i-1), 1, nnonzero - i + 1);
        i = nnonzero;
        %warning('Repeated atom detected. Algorithm stops.')
        break
    end
    idx_spcode = [idx_spcode idx];
    % Robust, correntropy-based regression
    [b, w] = CorrentropyReg(y, D(:, idx_spcode));
    X(idx_spcode,i) = b;
    r = sqrt(w).*(y - D(:, idx_spcode)*X(idx_spcode, i)); % Weighted residue
    E(:,i) = r;
end

X = X(:,1:i);
E = E(:,1:i);
x = X(:, end);
e = E(:, end);
end

%%
function [b, w] = CorrentropyReg(y, X)
% Robust correntropy-based regression
% Initialiazation based on OLS. Then, IRLS solution is iteratively updated
% until normalized L2-norm of difference of succesive solutions reaches a
% preset threshold.
%
% Parameters
% ----------
% y :   vector, size (m, 1)
%       Samples from dependent/response/measured variable in a linear 
%       regression framework
% X :   matrix, size (m, K)
%       Samples from K different regressors, input/independent variables in
%       a linear regression framework. For our problem, X is made up of K 
%       atoms from the dictionary D
% 
% Returns
% -------
% b :   vector, size (K, 1)
%       Effects or regression coefficients in a linear regression framework
% w :   vector, size (m, 1)
%       Weights associated to entries of y according to correntropy-based
%       linear regression

max_it = 100;                   % Maximum number of iterations
th = 1e-2;                      % Stopping criterion threshold for IRLS
inv_const = 0.00001;            % To avoid matrix-inversion-related errors
m = size(y,1);
[d, n] = size(X);
X2 = X'*X;                      % Compute to accelerate computations

% Initial estimate: ols
b = (X2 + inv_const*eye(n))\(X'*y);
e = y - X*b;
% Estimate sigma of gaussian kernel of correntropy
sig = sqrt((1/(2*m))*(sum(e.^2)));
% Fast calculation of matrix multiplications
JM = zeros(n,n,d);
for k = 1:d
    JM(:,:,k) = X(k,:)'*X(k,:);
end
JM = reshape(JM,n^2,d);

w = exp(-e.^2/(2*sig^2));
bprev = b;
it = 1;
fl = 1;
% IRLS - Iteratively Reweighted Least Squares
while fl
    Xmul = reshape(JM*w, n, n);
    b = (Xmul + inv_const*eye(n))\(bsxfun(@times,X,w)'*y);
    if sqrt(sum((b - bprev).^2))/sqrt(sum(bprev.^2)) <= th
        fl = 0;
    else
        % Compute values for weight vector
        e = y - X*b;
        sig = sqrt((1/(2*m))*(sum(e.^2)));
        w = exp(-e.^2/(2*sig^2));
        bprev = b;
        it = it + 1;
    end
    if it == max_it
        fl = 0;
        warning('Solution did not converge in maximum number of iterations allowed')
    end
end

end

