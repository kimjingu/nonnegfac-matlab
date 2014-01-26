function [ Z,numChol,numEq ] = normalEqComb( AtA,AtB,PassSet )
% Solve normal equations using combinatorial grouping.
% Although this function was originally adopted from the code of
% "M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450",
% important modifications were made to fix bugs.
%
% Modified by Jingu Kim (jingu.kim@gmail.com)
%             School of Computational Science and Engineering,
%             Georgia Institute of Technology
%
% Updated Aug-12-2009
% Updated Mar-13-2011: numEq,numChol
%
% numChol : number of unique cholesky decompositions done
% numEqs : number of systems of linear equations solved

	if isempty(AtB)
		Z = [];
		numChol = 0; numEq = 0;
	elseif (nargin==2) || all(PassSet(:))
        Z = AtA\AtB;
        numChol = 1; numEq = size(AtB,2);
    elseif size(AtA,1) ==1
        Z = AtB/AtA;
        numChol = 1; numEq = size(AtB,2);
	else
        Z = zeros(size(AtB));
        [n,k1] = size(PassSet);

        %% Fixed on Aug-12-2009
        if k1==1
			if any(PassSet)>0
            	Z(PassSet)=AtA(PassSet,PassSet)\AtB(PassSet); 
				numChol = 1; numEq = 1;
			else
				numChol = 0; numEq = 0;
			end
        else
            %% Fixed on Aug-12-2009
            % The following bug was identified by investigating a bug report by Hanseung Lee.
            % codedPassSet = 2.^(n-1:-1:0)*PassSet;
            % [sortedPassSet,sortIx] = sort(codedPassSet);
            % breaks = diff(sortedPassSet);
            % breakIx = [0 find(breaks) k1];

            [sortedPassSet,sortIx] = sortrows(PassSet');
            breaks = any(diff(sortedPassSet)');
            breakIx = [0 find(breaks) k1];

            %% Modified on Mar-11-2011
			% Skip columns with no passive sets
			if any(sortedPassSet(1,:))==0;
				startIx = 2;
			else
				startIx = 1;
			end
			numChol = 0; 
			numEq = k1-breakIx(startIx);

            for k=startIx:length(breakIx)-1
                cols = sortIx(breakIx(k)+1:breakIx(k+1));
				% Modified on Mar-13-2011
                % vars = PassSet(:,sortIx(breakIx(k)+1));
                vars = sortedPassSet(breakIx(k)+1,:)';
                Z(vars,cols) = AtA(vars,vars)\AtB(vars,cols);
                numChol = numChol + 1;
            end
        end
    end
end
