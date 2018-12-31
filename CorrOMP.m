function [X, E, w] = CorrOMP(y, D, K, err)
% Also known as CMP

if nargin == 3
    max_decomp = K;
    err = -1;
elseif nargin == 4
    max_decomp = 100;
end

r = y;
i = 0;
X = zeros(size(D,2), max_decomp);
E = zeros(length(y), max_decomp);
idx_as = [];
while (norm(r) > err && i < max_decomp)
    i = i + 1;
    abscorr = abs(r'*D);
    [~, idx] = max(abscorr);
    %idx_as = [idx_as idx];
    if isequal(idx_as, union(idx_as, idx))
        X(:,i:end) = repmat(X(:,i-1), 1, max_decomp - i + 1);
        E(:,i:end) = repmat(E(:,i-1), 1, max_decomp - i + 1);
        i = max_decomp;
        break
    end
    idx_as = union(idx_as, idx);
    [b, w] = CorrReg(y, D(:, idx_as));
    X(idx_as,i) = b;
    r = sqrt(w).*(y - D(:,idx_as)*X(idx_as,i));
    E(:,i) = r;
end

X = X(:,1:i);
E = E(:,1:i);

end

function [b, w] = CorrReg(y, X)

% X is the dictionary
max_it = 100;
th = 1e-2;
inv_const = 0.00001;
m = size(y,1);
[d, n] = size(X);

X2 = X'*X;

% Initial estimate: ols
b = (X2 + inv_const*eye(n))\(X'*y);
e = y - X*b;
% Estimate sigma
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
while fl
    % IRLS
    Xmul = reshape(JM*w, n, n);
    b = (Xmul + inv_const*eye(n))\(bsxfun(@times,X,w)'*y);
    if sqrt(sum(abs(b - bprev)).^2) <= th
        fl = 0;
    else
        % Compute values for weight function
        e = y - X*b;
        sig = sqrt((1/(2*m))*(sum(e.^2)));
        w = exp(-e.^2/(2*sig^2));
        bprev = b;
        it = it + 1;
    end
    if it == max_it
        fl = 0;
    end
end

end

