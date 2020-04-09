function y = AWGNoise(x, SNRdB)
% Add white gaussian noise (AWGN) with a given SNR (in dB)
% Part of RobOMP package. "RobOMP: Robust variants of Orthogonal Matching 
% Pursuit for sparse representations" DOI: 10.7717/peerj-cs.192 (open access)
% Author: Carlos Loza
% https://github.carlosloza/RobOMP
%
% Parameters
% ----------
% x :       vector, size (n, 1) or (1, n)
%           Input vector that will be distorted by AWGN
% SNRdB :   float (unsigned)
%           SNR goal (in dB) after AWGN distortion
%
% Returns
% -------
% y :       vector, size (n, 1) or (1, n). Same as x
%           Output vector after AWGN distortion

aux = var(x)/(10^(SNRdB/10));
Gnoise = randn(size(x));            % Gaussian noise
noise_add = sqrt(aux)*Gnoise;
y = x + noise_add;

end