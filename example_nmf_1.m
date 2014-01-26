% -------------------------------------------------------------------------
% Usage examples for nonnegative matrix factorization 
% -------------------------------------------------------------------------
m = 300;
n = 200;
k = 10;
A = rand(m,n);

% -------------------------------------------------------------------------
% Comment some of examples and try to execute!
% -------------------------------------------------------------------------
[W,H,iter,REC]=nmf(A,k,'tol',1e-3,'method','anls_bpp');
[W,H,iter,REC]=nmf(A,k,'verbose',2,'method','mu','max_iter',1000);
[W,H,iter,REC]=nmf(A,k,'verbose',1,'method','hals','max_iter',1000);

algs = {'anls_bpp' 'anls_asgroup' 'anls_asgivens' 'als' 'mu' 'hals'};
for i=1:length(algs)
    [W,H,iter,REC]=nmf(A,k,'verbose',1,'method',algs{i});
end

algs = {'anls_bpp' 'anls_asgroup' 'anls_asgivens' 'als' 'mu' 'hals'};
for i=1:length(algs)
    % Frobenius norm regularization test
    [W,H,iter,REC]=nmf(A,k,'tol',1e-3,'method',algs{i},'reg_w',[0.1 0],'reg_h',[0.8 0]);
    % L1-norm regularization test
    [W,H,iter,REC]=nmf(A,k,'tol',1e-3,'method',algs{i},'reg_w',[0.1 0],'reg_h',[0 0.8]);
end
