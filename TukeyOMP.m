function [X, E, w] = TukeyOMP(y, D, K, err)

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
    [b, w] = TukeyReg(y, D(:, idx_as));
    X(idx_as,i) = b;
    %r = sqrt(w).*(y - D(:,idx_as)*X(idx_as,i));
    r = w.*(y - D(:,idx_as)*X(idx_as,i));
    E(:,i) = r;
end

X = X(:,1:i);
E = E(:,1:i);

end

function [b, w] = TukeyReg(y, X)

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
% Estimate scale
s = 1.4824*median(abs(e - median(e)));
res = e/s;
w = zeros(length(e),1);
idx1 = abs(res) < 4.685;
w(idx1) = (1 - (res(idx1)/4.685).^2).^2;

% Fast calculation of matrix multiplications
JM = zeros(n,n,d);
for k = 1:d
    JM(:,:,k) = X(k,:)'*X(k,:);
end
JM = reshape(JM,n^2,d);

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
        s = 1.4824*median(abs(e - median(e)));
        res = e/s;
        w = zeros(length(e),1);
        idx1 = abs(res) < 4.685;
        w(idx1) = (1 - (res(idx1)/4.685).^2).^2;
        
        %inv_const = n*s/(b'*b);
        
        bprev = b;
        it = it + 1;
    end
    if it == max_it
        fl = 0;
    end
end

end
