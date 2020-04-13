function [x, e, w, X, E] = RobOMP(y, D, varargin)
% Implementation of Robust variants of Orthogonal Matching Pursuit based on
% M-estimators
% Author: Carlos Loza
% Part of RobOMP package. DOI: 10.7717/peerj-cs.192 (open access)
% https://github.carlosloza/RobOMP
%
% Parameters
% ----------
% y :           vector, size (m, 1)
%               Signal to be sparsely encoded
% D :           matrix, size (m , n)
%               Dictionary/measurement matrix made up of atoms
% warmstart :   int
%               If set to 1, the algorithm uses the Huber solution as
%               initialization 
%               Default: 1
% m-est :       string
%               M-estimator for linear regression subroutine of Orthogonal
%               Matching Pursuit. Cases: Cauchy, Fair, Huber, Tukey, Welsch
%               Default: Huber
% c :           float
%               Hyperparameter of M-estimators
%               Defaults are set on a per-estimator basis according to 95% 
%               asymptotic efficiency on the standar Normal distribution, 
%               see Table 2 of Loza 2019 for specifics
% nnonzero :    int
%               Number of non-zero coefficients in sparse code
%               This is equal to number of iterations in Orthogonal Matching
%               Pursuit (OMP)
% tol :         float
%               Residual norm tolerance. Dispersion/power rate not explained
%               by the sparse code with respect to the norm of y
%               Default: 0.1 i.e. 10% of the L2 norm of input y
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
% Example: x = RobOMP(y, D, 'warmstart', 0, 'm-est', 'Tukey', 'nnonzero', 10, 'tol', 0.25)

warmst = 1;                     % default warmstart
[m, n] = size(D);
% Check inputs
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'warmstart')
        warmst = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'm-est')
        m_est = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'c')
        m_est_hyperp = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'nnonzero')
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
% Default m-estimator
if ~exist('m_est', 'var')
    m_est = 'Tukey';
end
if ~exist('m_est_hyperp', 'var')
     % Optimal default hyperparameters
    if strcmpi(m_est, 'Cauchy')
        m_est_hyperp = 2.385;
    elseif strcmpi(m_est, 'Fair')
        m_est_hyperp = 1.4;
    elseif strcmpi(m_est, 'Huber')
        m_est_hyperp = 1.345;
    elseif strcmpi(m_est, 'Tukey')
        m_est_hyperp = 4.685;
    elseif strcmpi(m_est, 'Welsch')
        m_est_hyperp = 2.985;
    end
end

switch flcase
    case 1
        tol = -1;
    case 2
        nnonzero = n;           % This depends on the problem
    case 3
        nnonzero = n;           % This depends on the problem
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
        warning('Repeated atom detected. Algorithm stops.')
        break
    end
    idx_spcode = [idx_spcode idx];
    % Robust, m-estimator-based regression
    [b, w] = RobReg(y, D(:, idx_spcode), warmst, m_est, m_est_hyperp);
    X(idx_spcode, i) = b;
    r = w.*(y - D(:, idx_spcode)*X(idx_spcode, i));           % Weighted residue
    E(:,i) = r;
end

X = X(:,1:i);
E = E(:,1:i);
x = X(:, end);
e = E(:, end);
end

%%
function [b, w] = RobReg(y, X, warmst, m_est, m_est_hyperp)
% Wrapper for  M-estimator-based regression
% Initialiazation based on Huber variant. Then, IRLS solution is iteratively 
% updated until normalized L2-norm of difference of succesive solutions 
% reaches a preset threshold. Huber solution is initilized with OLS solution
%
% Parameters
% ----------
% y :               vector, size (m, 1)
%                   Samples from dependent/response/measured variable, 
%                   in a linear regression framework
% X :               matrix, size (m, K)
%                   Samples from K different regressors, input/independent
%                   variables in a linear regression framework. For our 
%                   problem, X is made up of K atoms from the dictionary D
% warmst :          Same as main function
% m_est :           Same as main function
% m_est_hyperp :    Same as main function
% 
% Returns
% -------
% b :   vector, size (K, 1)
%       Effects or regression coefficients in a linear regression framework
% w :   vector, size (m, 1)
%       Weights associated to entries of y according to correntropy-based
%       linear regression

if warmst
    % Warm start
    % Initial robust solution: Huber variant
    [b, ~] = MEstimator(y, X, 'Huber', 1.345);
    % Solve for selected m-estimator
    [b, w] = MEstimator(y, X, m_est, m_est_hyperp, b);
else
    % Cold start
    [b, w] = MEstimator(y, X, m_est, m_est_hyperp);
end

end

%%
function [b, w] = MEstimator(y, X, m_est, m_est_hyperp, b)
% Actual M-estimator-based regression
% Same Parameters and Returns as wrapper
% Extra parameter b is the initial solution (from Huber variant)

% Handles for m-estimator functions
if strcmpi(m_est, 'Cauchy')
    fm_est = str2func('@Cauchy');
elseif strcmpi(m_est, 'Fair')
    fm_est = str2func('@Fair');
elseif strcmpi(m_est, 'Huber')
    fm_est = str2func('@Huber');
elseif strcmpi(m_est, 'Tukey')
    fm_est = str2func('@Tukey');
elseif strcmpi(m_est, 'Welsch')
    fm_est = str2func('@Welsch');
end

max_it = 100;                   % Maximum number of iterations
th = 1e-2;                      % Stopping criterion threshold for IRLS
inv_const = 0.00001;            % To avoid matrix-inversion-related errors
[d, n] = size(X);

% If no initial solution is provided, use ols
if nargin == 4
    X2 = X'*X;                      % Compute to accelerate computations
    b = (X2 + inv_const*eye(n))\(X'*y);
end
e = y - X*b;
% Estimate scale
s = 1.4824*median(abs(e - median(e)));
w = fm_est(e, s, m_est_hyperp);
% Fast calculation of matrix multiplications
JM = zeros(n,n,d);
for k = 1:d
    JM(:,:,k) = X(k,:)'*X(k,:);
end
JM = reshape(JM,n^2,d);

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
        % Compute values for weight function
        e = y - X*b;
        s = 1.4824*median(abs(e - median(e)));
        w = fm_est(e, s, m_est_hyperp);        
        bprev = b;
        it = it + 1;
    end
    if it == max_it
        fl = 0;
        warning('Solution did not converge in maximum number of iterations allowed')
    end
end

end

%% Cauchy m-estimator weighting
function w = Cauchy(e, s, m_est_hyperp)
res = e/(m_est_hyperp*s);
w = 1./(1 + res.^2);
end

%% Fair m-estimator weighting
function w = Fair(e, s, m_est_hyperp)
res = e/(m_est_hyperp*s);
w = 1./(1 + abs(res));
end

%% Huber m-estimator weighting
function w = Huber(e, s, m_est_hyperp)
res = e/s;
w = ones(length(e),1);
idx1 = abs(res) >= m_est_hyperp;
w(idx1) = m_est_hyperp./abs(res(idx1));
end

%% Tukey m-estimator weighting
function w = Tukey(e, s, m_est_hyperp)
res = e/s;
w = zeros(length(e),1);
idx1 = abs(res) < m_est_hyperp;
w(idx1) = (1 - (res(idx1)/m_est_hyperp).^2).^2;
end

%% Welsch m-estimator weighting
function w = Welsch(e, s, m_est_hyperp)
res = e/(m_est_hyperp*s);
w = exp(-(res.^2));
end