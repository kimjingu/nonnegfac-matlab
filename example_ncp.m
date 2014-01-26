% -----------------------------------------------
% Creating a synthetic 4th-order tensor
% -----------------------------------------------
N1=20;
N2=25;
N3=30;
N4=30;

R=10;

A_org = rand(N1,R); A_org( A_org < 0.4 ) = 0;
B_org = rand(N2,R); B_org( B_org < 0.4 ) = 0;
C_org = rand(N3,R); C_org( C_org < 0.4 ) = 0;
D_org = rand(N4,R); D_org( D_org < 0.4 ) = 0;

X_ks = ktensor({A_org,B_org,C_org,D_org});
X_ks = arrange(X_ks);
X = full(ktensor(X_ks));

% -----------------------------------------------
% Tentative initial values
% -----------------------------------------------
A0 = rand(N1,R);
B0 = rand(N2,R);
C0 = rand(N3,R);
D0 = rand(N4,R);

Finit = cell(4,1);
Finit{1}=A0;
Finit{2}=B0;
Finit{3}=C0;
Finit{4}=D0;

% -----------------------------------------------
% Uncomment only one of the following
% -----------------------------------------------
[X_approx_ks,iter,REC] = ncp(X,R);
% [X_approx_ks,iter,REC] = ncp(X,R,'method','hals');
% [X_approx_ks,iter,REC] = ncp(X,R,'method','anls_asgroup');
% [X_approx_ks,iter,REC] = ncp(X,R,'method','mu');
% [X_approx_ks,iter,REC] = ncp(X,R,'tol',1e-7,'max_iter',300);
% [X_approx_ks,iter,REC] = ncp(X,R,'tol',1e-7,'max_iter',300,'verbose',2);
% [X_approx_ks,iter,REC] = ncp(X,R,'init',Finit);
% [X_approx_ks,iter,REC] = ncp(X,R,'init',Finit,'verbose',1);

% -----------------------------------------------
% Approximation Error and Factor Errors
% Note: Factor errors maybe sometimes large due to the reordering of columns. 
%       A more appropriate way would be first find the correct reordering
%       using, e.g., the Hungradian algorithm.
% -----------------------------------------------
X_approx = full(X_approx_ks);
X_err = norm(X-X_approx)/norm(X);
X_err

A_err = norm(X_ks.U{1}-X_approx_ks.U{1})/norm(X_ks.U{1});
B_err = norm(X_ks.U{2}-X_approx_ks.U{2})/norm(X_ks.U{2});
C_err = norm(X_ks.U{3}-X_approx_ks.U{3})/norm(X_ks.U{3});
D_err = norm(X_ks.U{4}-X_approx_ks.U{4})/norm(X_ks.U{4});
A_err
B_err
C_err
D_err
