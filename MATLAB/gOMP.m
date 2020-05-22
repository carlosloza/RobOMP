function [x, e, X, E, Xext, Eext] = gOMP(y, D, varargin)
% Implementation of Generalized Orthogonal Matching Pursuit (gOMP) as
% introduced by Wang, Kwon, and Shim 2011 (DOI: 10.1109/TSP.2012.2218810)
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
% N0 :      int, optional
%           Number of atoms chosen per iteration
%           Default: 1, i.e. regular orthogonal matching pursuit (OMP)
% nnonzero :int, optional
%           Number of non-zero coefficients in sparse code
%           If nnonzero is not a multiple of N0, a warning flag is displayed
%           and the actual number of non-zero coefficients in the sparse
%           code is set to N0*floor(nnonzero/N0)
%           nnonzero is equal to the number of iterations (i.e. OMP case),
%           only when N0 = 1.
%           Default: n, i.e extreme non-sparse case
% tol :     float, optional
%           Residual norm tolerance. Dispersion/power rate not explained by
%           the sparse code with respect to the norm of y
%           Default: 0.1 i.e. 10% of the L2 norm of input y
%
% If neither nnonzero nor tol are set, then tol is set to default
% If both nnonzero and tol are set, then the algorithm stops when both 
% conditions are met
%
% Returns
% -------
% x :       vector, size (n, 1)
%           Sparse code corresponding to y encoded by D with sparsity level
%           nnonzero
% e :       vector, size (m, 1)
%           Residue/error after sparse coding of y with sparsity level
%           nnonzero
% X :       matrix, size (n, maxiter)
%           Same as x, but each column corresponds to decreasingly sparser
%           solutions according to maxiter
% E :       matrix, size (m, maxiter)
%           Same as e, but each column corresponds to residue after  
%           decreasingly sparser solutions, i.e. likewise X
% Xext :    matrix, size (n, nnonzero)
%           same as x, but each column corresponds to decreasingly sparser
%           solutions according to the number of gOMP iterations. nnonzero
%           is the original sparsity set by the user. This matrix will have
%           repeated inputs/columns if N0 ~= 0. It is mainly used for
%           comparisons with classic OMP encoders where N0 = 1
% Eext :    matrix, size (m, nnonzero)
%           Same as e, but each column corresponds to residue after  
%           decreasingly sparser solutions, i.e. likewise Xext
%
% Example: x = gOMP(y, D, 'N0', 2, 'nnonzero', 10, 'tol', 0.25)

N0 = 1;      % Default number of atoms chosen per iteration, i.e. OMP
[m, n] = size(D);
% Check inputs
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'nnonzero')
        nnonzero = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'tol')
        tol = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'N0')
        N0 = varargin{i + 1};
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

nnonzero_ini = nnonzero;
% Check if N0 is larger than nnonzero and if nnonzero is multiple of N0
if N0 > nnonzero
    %     %warning('N0 is larger than nnonzero. N0 will be set to nnonzero')
    N0 = nnonzero;           % Extreme case, only one iteration is needed
    nnonzero_ini = N0;
end
if mod(nnonzero, N0) ~= 0
%     warning(['nnonzero is not a multiple of N0. Actual support of sparse code ', ...
%         'will be decreased'])
    nnonzero = N0*floor(nnonzero/N0);
end

maxiter = nnonzero/N0;
normy = norm(y);
r = y;
i = 0;
X = zeros(n, maxiter);
E = zeros(m, maxiter);
idx_spcode = [];
while (norm(r)/normy > tol && i < maxiter)
    i = i + 1;
    abscorr = abs(r'*D);
    [~, idx] = sort(abscorr, 'descend');
    if any(diff(sort([idx_spcode idx(1:N0)])) == 0)  % Case for repeated atoms
        X(:,i:end) = repmat(X(:,i-1), 1, maxiter - i + 1);
        E(:,i:end) = repmat(E(:,i-1), 1, maxiter - i + 1);
        i = maxiter;
        warning('Repeated atom detected. Algorithm stops.')
        break
    end
    idx_spcode = [idx_spcode idx(1:N0)];
    % Ordinary least-squares (OLS) regression
    b = D(:, idx_spcode)\y;
    X(idx_spcode, i) = b;
    r = y - D(:, idx_spcode)*X(idx_spcode, i);    % Residue
    E(:,i) = r;
end

X = X(:, 1:i);
E = E(:, 1:i);
x = X(:, end);
e = E(:, end);

% Extended versions for comparisons with classic OMP encoders
Xext = zeros(n, nnonzero);
Eext = zeros(m, nnonzero);
ct = 1;
for i = 1:N0:nnonzero
    Xext(:, i:i + N0 - 1) = repmat(X(:, ct), 1, N0);
    Eext(:, i:i + N0 - 1) = repmat(E(:, ct), 1, N0);
    ct = ct + 1;
end
if nnonzero_ini > nnonzero
    Xext = [Xext repmat(X(:, end), 1, nnonzero_ini - nnonzero)];
    Eext = [Eext repmat(E(:, end), 1, nnonzero_ini - nnonzero)];
end

end
