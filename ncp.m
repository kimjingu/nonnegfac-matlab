% Nonnegative Tensor Factorization (Canonical Decomposition / PARAFAC)
%
% Written by Jingu Kim (jingu.kim@gmail.com)
%            School of Computational Science and Engineering,
%            Georgia Institute of Technology
%
% This software implements nonnegativity-constrained low-rank approximation of tensors in PARAFAC model.
% Assuming that a k-way tensor X and target rank r are given, this software seeks F1, ... , Fk 
% by solving the following problem:
%
%     minimize  || X- sum_(j=1)^r (F1_j o F2_j o ... o Fk_j) ||_F^2 + G(F1, ... , Fk) + H(F1, ..., Fk)
%     where
%           G(F1, ... , Fk) = sum_(i=1)^k ( alpha_i * ||Fi||_F^2 ),
%           H(F1, ... , Fk) = sum_(i=1)^k ( beta_i sum_(j=1)^n || Fi_j ||_1^2 ).
%     such that
%           Fi >= 0 for all i.
%
% To use this software, it is necessary to first install MATLAB Tensor Toolbox
% by Brett W. Bader and Tamara G. Kolda, available at http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox/.
% The latest version that was tested with this software is Version 2.4, March 2010.
% Refer to the help manual of the toolbox for installation and basic usage.
%
% Reference:
%      Jingu Kim and Haesun Park.
%      Fast Nonnegative Tensor Factorization with an Active-set-like Method.
%      In High-Performance Scientific Computing: Algorithms and Applications, Springer, 2012, pp. 311-326.
%
% Please send bug reports, comments, or questions to Jingu Kim.
% This code comes with no guarantee or warranty of any kind.
%
% Last modified 03/26/2012
%
% <Inputs>
%        X : Input data tensor. X is a 'tensor' object of tensor toolbox.
%        r : Target low-rank
%
%        (Below are optional arguments: can be set by providing name-value pairs)
%
%        METHOD : Algorithm for solving NMF. One of the following values:
%                 'anls_bpp' 'anls_asgroup' 'hals' 'mu'
%                 See above paper (and references therein) for the details of these algorithms.
%                 Default is 'anls_bpp'.
%        TOL : Stopping tolerance. Default is 1e-4. If you want to obtain a more accurate solution, 
%               decrease TOL and increase MAX_ITER at the same time.
%        MIN_ITER : Minimum number of iterations. Default is 20.
%        MAX_ITER : Maximum number of iterations. Default is 200.
%        INIT : A cell array that contains initial values for factors Fi.
%               See examples to learn how to set.
%        VERBOSE : 0 (default) - No debugging information is collected.
%                  1 (debugging/experimental purpose) - History of computation is returned. See 'REC' variable.
%                  2 (debugging/experimental purpose) - History of computation is additionally printed on screen.
% <Outputs>
%        F : a 'ktensor' object that represent a factorized form of a tensor. See tensor toolbox for more info.
%        iter : Number of iterations
%        REC : (debugging/experimental purpose) Auxiliary information about the execution
% <Usage Examples> 
%        F = ncpp(X,5);
%        F = ncp(X,10,'tol',1e-3);
%        F = ncp(X,10,'tol',1e-3,'verbose',2);
%        F = ncp(X,7,'init',Finit,'tol',1e-5,'verbose',2);

function [F,iter,REC]=ncp(X,r,varargin)
    % set parameters
    params = inputParser;
    params.addParamValue('method'           ,'anls_bpp' ,@(x) ischar(x) );
    params.addParamValue('tol'              ,1e-4       ,@(x) isscalar(x) & x > 0 );
    params.addParamValue('stop_criterion'   ,1          ,@(x) isscalar(x) & x >= 0);
    params.addParamValue('min_iter'         ,20         ,@(x) isscalar(x) & x > 0);
    params.addParamValue('max_iter'         ,200        ,@(x) isscalar(x) & x > 0 );
    params.addParamValue('max_time'         ,1e6        ,@(x) isscalar(x) & x > 0);
    params.addParamValue('init'             ,cell(0)    ,@(x) iscell(x) );
    params.addParamValue('verbose'          ,0          ,@(x) isscalar(x) & x >= 0 );
    params.addParamValue('orderWays',[]);
    params.parse(varargin{:});
    
    % copy from params object
    par = params.Results;
    par.nWay = ndims(X);
    par.r = r;
    par.size = size(X);

    if isempty(par.orderWays)
        par.orderWays = [1:par.nWay]; 
    end

    % set initial values
    if ~isempty(par.init)
        F_cell = par.init;
        par.init_type = 'User provided';
        par.init = cell(0);
    else
        Finit = cell(par.nWay,1);
        for i=1:par.nWay
            Finit{i}=rand(size(X,i),r);
        end
        F_cell = Finit;
        par.init_type = 'Randomly generated';
    end

    % This variable is for analysis/debugging, so it does not affect the output (W,H) of this program
    REC = struct([]);
    tPrev = cputime;
    REC(1).start_time = datestr(now);
    grad = getGradient(X,F_cell,par);
    ver= struct([]);

    clear('init');
    init.nr_X = norm(X);
    init.nr_grad_all = 0;
    for i=1:par.nWay
        this_value = norm(grad{i},'fro');
        init.(['nr_grad_',num2str(i)])  = this_value;
        init.nr_grad_all = init.nr_grad_all + this_value^2;
    end
    init.nr_grad_all = sqrt(init.nr_grad_all);
    REC(1).init = init;

    initializer= str2func([par.method,'_initializer']);
    iterSolver = str2func([par.method,'_iterSolver']);
    iterLogger = str2func([par.method,'_iterLogger']);

    % Collect initial information for analysis/debugging
    if par.verbose          
        tTemp = cputime;
        prev_F_cell = F_cell;
        pGrad = getProjGradient(X,F_cell,par);
        ver = prepareHIS(ver,X,F_cell,ktensor(F_cell),prev_F_cell,pGrad,init,par,0,0);
        tPrev = tPrev+(cputime-tTemp);
    end

    % Execute initializer
    [F_cell,par,val,ver] = feval(initializer,X,F_cell,par,ver);

    if par.verbose & ~isempty(ver)
        tTemp = cputime;
        if par.verbose == 2, display(ver);, end
        REC.HIS = ver;
        tPrev = tPrev+(cputime-tTemp);
    end

    REC(1).par = par;
    tTemp = cputime; display(par); tPrev = tPrev+(cputime-tTemp);
    tStart = tPrev;, tTotal = 0; 

    if (par.stop_criterion == 2) && ~isfield(ver,'rel_Error')
        F_kten = ktensor(F_cell);
        ver(1).rel_Error = getRelError(X,ktensor(F_cell),init);
    end

    % main iterations
    for iter=1:par.max_iter;
        cntu = 1;

        [F_cell,val] = feval(iterSolver,X,F_cell,iter,par,val);
        pGrad = getProjGradient(X,F_cell,par);
        F_kten = ktensor(F_cell);

        prev_Ver = ver;
        ver= struct([]);
        if (iter >= par.min_iter)
            if (par.verbose && (tTotal > par.max_time)) || (~par.verbose && ((cputime-tStart)>par.max_time))
                cntu = 0;
            else
                switch par.stop_criterion
                    case 1
                        ver(1).SC_PGRAD = getStopCriterion(pGrad,init,par);
                        if (ver.SC_PGRAD<par.tol) cntu = 0; end
                    case 2
                        ver(1).rel_Error = getRelError(X,F_kten,init);
                        ver.SC_DIFF = abs(prev_Ver.rel_Error - ver.rel_Error);
                        if (ver.SC_DIFF<par.tol) cntu = 0; end
                    case 99 
                        ver(1).rel_Error = getRelError(X,F_kten,init);
                        if ver(1).rel_Error< 1 cntu = 0; end
                end
            end
        end

        % Collect information for analysis/debugging
        if par.verbose          
            elapsed = cputime-tPrev;
            tTotal = tTotal + elapsed;

            ver = prepareHIS(ver,X,F_cell,F_kten,prev_F_cell,pGrad,init,par,iter,elapsed);
            ver = feval(iterLogger,ver,par,val,F_cell,prev_F_cell);
            if ~isfield(ver,'SC_PGRAD')
                ver.SC_PGRAD = getStopCriterion(pGrad,init,par);
            end
            if ~isfield(ver,'SC_DIFF')
                ver.SC_DIFF = abs(prev_Ver.rel_Error - ver.rel_Error);
            end
            REC.HIS = saveHIS(iter+1,ver,REC.HIS);
            prev_F_cell = F_cell;
            if par.verbose == 2, display(ver);, end
            tPrev = cputime;
        end
        
        if cntu==0, break; end
    end
    F = arrange(F_kten);

    % print finishing information
    final.iterations = iter;
    if par.verbose
        final.elapsed_sec = tTotal;
    else
        final.elapsed_sec = cputime-tStart;
    end
    for i=1:par.nWay
        final.(['f_density_',num2str(i)])   = length(find(F.U{i}>0))/(size(F.U{i},1)*size(F.U{i},2));
    end
    final.rel_Error = getRelError(X,F_kten,init);
    REC.final = final;
    
    REC.finish_time = datestr(now);

    display(final);
end

%----------------------------------------------------------------------------------------------
%                                    Utility Functions 
%----------------------------------------------------------------------------------------------
function ver = prepareHIS(ver,X,F,F_kten,prev_F,pGrad,init,par,iter,elapsed)
    ver(1).iter             = iter;
    ver.elapsed             = elapsed;
    if ~isfield(ver,'rel_Error')
        ver.rel_Error       = getRelError(X,F_kten,init);
    end
    for i=1:par.nWay
        ver.(['f_change_',num2str(i)])  = norm(F{i}-prev_F{i});
        ver.(['f_density_',num2str(i)]) = length(find(F{i}>0))/(size(F{i},1)*size(F{i},2));
        ver.(['rel_nr_pgrad_',num2str(i)])  = norm(pGrad{i},'fro')/init.(['nr_grad_',num2str(i)]);
    end
end

function HIS = saveHIS(idx,ver,HIS)
    fldnames = fieldnames(ver);

    for i=1:length(fldnames)
        flname = fldnames{i};
        HIS.(flname)(idx) = ver.(flname);
    end
end

function rel_Error = getRelError(X,F_kten,init)
    rel_Error = sqrt(max(init.nr_X^2 + norm(F_kten)^2 - 2 * innerprod(X,F_kten),0))/init.nr_X;
end

function [grad] = getGradient(X,F,par)
    grad = cell(par.nWay,1);
    for k=1:par.nWay
        ways = 1:par.nWay;
        ways(k)='';
        XF = mttkrp(X,F,k);
        % Compute the inner-product matrix
        FF = ones(par.r,par.r);
        for i = ways
            FF = FF .* (F{i}'*F{i});
        end
        grad{k} = F{k} * FF - XF;
    end
end

function [pGrad] = getProjGradient(X,F,par)
    pGrad = cell(par.nWay,1);
    for k=1:par.nWay
        ways = 1:par.nWay;
        ways(k)='';
        XF = mttkrp(X,F,k);
        % Compute the inner-product matrix
        FF = ones(par.r,par.r);
        for i = ways
            FF = FF .* (F{i}'*F{i});
        end
        grad = F{k} * FF - XF;
        pGrad{k} = grad(grad<0|F{k}>0);
    end
end

function retVal = getStopCriterion(pGrad,init,par)
    retVal = 0;
    for i=1:par.nWay
        retVal = retVal + (norm(pGrad{i},'fro'))^2;
    end
    retVal = sqrt(retVal)/init.nr_grad_all;
end

% 'anls_bpp' : ANLS with Block Principal Pivoting Method 
% Reference:
%    Jingu Kim and Haesun Park.
%    Fast Nonnegative Tensor Factorization with an Active-set-like Method.
%    In High-Performance Scientific Computing: Algorithms and Applications, 
%    Springer, 2012, pp. 311-326.
function [F,par,val,ver] = anls_bpp_initializer(X,F,par,ver)
    F{par.orderWays(1)} = zeros(size(F{par.orderWays(1)}));

    for k=1:par.nWay
        ver(1).(['turnZr_',num2str(k)]) = 0;
        ver.(['turnNz_',num2str(k)])    = 0;
        ver.(['numChol_',num2str(k)])   = 0;
        ver.(['numEq_',num2str(k)])     = 0;
        ver.(['suc_',num2str(k)])       = 0;
    end
    val.FF = cell(par.nWay,1);
    for k=1:par.nWay
        val.FF{k} = F{k}'*F{k};
    end
end

function [F,val] = anls_bpp_iterSolver(X,F,iter,par,val)
    % solve NNLS problems for each factor
    for k=1:par.nWay
        curWay = par.orderWays(k);
        ways = 1:par.nWay;
        ways(curWay)='';
        XF = mttkrp(X,F,curWay);
        % Compute the inner-product matrix
        FF = ones(par.r,par.r);
        for i = ways
            FF = FF .* val.FF{i};
        end
        [Fthis,temp,sucThis,numCholThis,numEqThis] = nnlsm_blockpivot(FF,XF',1,F{curWay}');
        F{curWay}=Fthis';
        val(1).FF{curWay} = F{curWay}'*F{curWay};
        val.(['numChol_',num2str(k)]) = numCholThis;
        val.(['numEq_',num2str(k)])     = numEqThis;
        val.(['suc_',num2str(k)])       = sucThis;
    end
end

function [ver] = anls_bpp_iterLogger(ver,par,val,F,prev_F)
    for k=1:par.nWay
        ver.(['turnZr_',num2str(k)])    = length(find( (prev_F{k}>0) & (F{k}==0) ))/(size(F{k},1)*size(F{k},2));
        ver.(['turnNz_',num2str(k)])    = length(find( (prev_F{k}==0) & (F{k}>0) ))/(size(F{k},1)*size(F{k},2));
        ver.(['numChol_',num2str(k)])   = val.(['numChol_',num2str(k)]);
        ver.(['numEq_',num2str(k)])     = val.(['numEq_',num2str(k)]);
        ver.(['suc_',num2str(k)])       = val.(['suc_',num2str(k)]);
    end
end

% 'anls_asgroup' : ANLS with Active Set Method and Column Grouping
% Reference:
%    Kim, H. and Park, H. and Elden, L.
%    Non-negative Tensor Factorization Based on Alternating Large-scale Non-negativity-constrained Least Squares.
%    In Proceedings of IEEE 7th International Conference on Bioinformatics and Bioengineering 
%    (BIBE07), 2, pp. 1147-1151,2007
function [F,par,val,ver] = anls_asgroup_initializer(X,F,par,ver)
    [F,par,val,ver] = anls_bpp_initializer(X,F,par,ver);
end

function [F,val] = anls_asgroup_iterSolver(X,F,iter,par,val)
    % solve NNLS problems for each factor
    for k=1:par.nWay
        curWay = par.orderWays(k);
        ways = 1:par.nWay;
        ways(curWay)='';
        XF = mttkrp(X,F,curWay);
        % Compute the inner-product matrix
        FF = ones(par.r,par.r);
        for i = ways
            FF = FF .* val.FF{i};
        end
        ow = 0;
        [Fthis,temp,sucThis,numCholThis,numEqThis] = nnlsm_activeset(FF,XF',ow,1,F{curWay}');
        F{curWay}=Fthis';
        val(1).FF{curWay} = F{curWay}'*F{curWay};
        val.(['numChol_',num2str(k)]) = numCholThis;
        val.(['numEq_',num2str(k)])     = numEqThis;
        val.(['suc_',num2str(k)])       = sucThis;
    end
end

function [ver] = anls_asgroup_iterLogger(ver,par,val,F,prev_F)
    ver = anls_bpp_iterLogger(ver,par,val,F,prev_F);
end

% 'mu' : Multiplicative Updating Method
% Reference:
%    M. Welling and M. Weber.
%    Positive tensor factorization.
%    Pattern Recognition Letters, 22(12), pp. 1255–1261, 2001.
function [F,par,val,ver] = mu_initializer(X,F,par,ver)
    val.FF = cell(par.nWay,1);
    for k=1:par.nWay
        val.FF{k} = F{k}'*F{k};
    end
end

function [F,val] = mu_iterSolver(X,F,iter,par,val)
    epsilon = 1e-16;

    for k=1:par.nWay
        curWay = par.orderWays(k);
        ways = 1:par.nWay;
        ways(curWay)='';
        % Calculate Fnew = X_(n) * khatrirao(all U except n, 'r').
        XF = mttkrp(X,F,curWay);
        % Compute the inner-product matrix
        FF = ones(par.r,par.r);
        for i = ways
            FF = FF .* val.FF{i};
        end
        F{curWay} = F{curWay}.*XF./(F{curWay}*FF+epsilon);
        val(1).FF{curWay} = F{curWay}'*F{curWay};
    end
end

function [ver] = mu_iterLogger(ver,par,val,F,prev_F)
end

% 'hals' : Hierarchical Alternating Least Squares Method
% Reference:
%    Cichocki, A. and Phan, A.H.
%    Fast local algorithms for large scale nonnegative matrix and tensor factorizations.
%    IEICE Trans. Fundam. Electron. Commun. Comput. Sci. E92-A(3), 708–721 (2009)
function [F,par,val,ver] = hals_initializer(X,F,par,ver)
    % normalize
    d = ones(1,par.r);
    for k=1:par.nWay-1
        curWay = par.orderWays(k);
        norm2 = sqrt(sum(F{curWay}.^2,1));
        F{curWay} = F{curWay}./repmat(norm2,size(F{curWay},1),1);
        d = d .* norm2;
    end
    curWay = par.orderWays(end);
    F{curWay} = F{curWay}.*repmat(d,size(F{curWay},1),1);

    val.FF = cell(par.nWay,1);
    for k=1:par.nWay
        val.FF{k} = F{k}'*F{k};
    end
end

function [F,val] = hals_iterSolver(X,F,iter,par,val)
    epsilon = 1e-16;

    d = sum(F{par.orderWays(end)}.^2,1);

    for k=1:par.nWay
        curWay = par.orderWays(k);
        ways = 1:par.nWay;
        ways(curWay)='';
        % Calculate Fnew = X_(n) * khatrirao(all U except n, 'r').
        XF = mttkrp(X,F,curWay);
        % Compute the inner-product matrix
        FF = ones(par.r,par.r);
        for i = ways
            FF = FF .* val.FF{i};
        end
        if k<par.nWay
            for j = 1:par.r
                F{curWay}(:,j) = max( d(j) * F{curWay}(:,j) + XF(:,j) - F{curWay} * FF(:,j),epsilon);
                F{curWay}(:,j) = F{curWay}(:,j) ./ norm(F{curWay}(:,j));
            end
        else
            for j = 1:par.r
                F{curWay}(:,j) = max( F{curWay}(:,j) + XF(:,j) - F{curWay} * FF(:,j),epsilon);
            end
        end
        val(1).FF{curWay} = F{curWay}'*F{curWay};
    end
end

function [ver] = hals_iterLogger(ver,par,val,F,prev_F)
end
