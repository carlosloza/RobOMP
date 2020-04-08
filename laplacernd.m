function y  = laplacernd(m, n, mu, s)
% Implementation of random number generator according to Laplace distribution
% Author: Carlos Loza
% Part of RobOMP package. "RobOMP: Robust variants of Orthogonal Matching 
% Pursuit for sparse representations" DOI: 10.7717/peerj-cs.192 (open access)
% https://github.carlosloza/RobOMP
%
% Parameters
% ----------
% m :       int
%           First dimension of generated samples, i.e. rows
% n :       int
%           Second dimension of generated samples, i.e. columns
% mu :      float
%           Mean of Laplace distribution
%           Default: 0
% s :       float (unsigned)
%           Standard deviation of Laplace distribution
%           Default: 1
%
% Returns
% -------
% y :       matrix, size (m, n)
%           i.i.d. randomly generated samples according to Laplace
%           distribution

%Check inputs
switch nargin
    case 2
        mu = 0;
        s = 1;
    case 3
        s = 1;
end

% Inverse transform sampling
u = rand(m, n) - 0.5;                       % Uniform random numbers
a = s / sqrt(2);
y = mu - a * sign(u).* log(1- 2* abs(u));
end