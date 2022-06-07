FlagUseSemiknown = true;
FlagCalculateDuals = false;

diary(strcat('myMatlabDiaryFile.txt'));

load('inflationMomentMat.mat');
load('inflationKnownMoments.mat');
load('inflationProptoMoments.mat');

nr_unknown_moments = double(max(G(:)) - length(known_moments));
freevars = sdpvar(1, nr_unknown_moments);
lambda = sdpvar(1);
slots = [known_moments, freevars];
G = slots(G);
constraint_maxmineig = [ G - lambda*eye(size(G)) >= 0 ];
constraints_semiknown = [];
if FlagUseSemiknown
	for i=1:size(propto,1)
	   var1 = slots(propto(i,1));
	   coeff = propto(i,2);
	   var2 = slots(propto(i,3));
	   constraints_semiknown = [constraints_semiknown, var1 == coeff * var2];
	end
end
constraint = [constraint_maxmineig, constraints_semiknown];
clearvars slots nr_unknown_moments;
sol = optimize(constraint,-lambda,...
    sdpsettings(...
        'solver','mosek',...
        'verbose',1,'dualize',0,...
        'showprogress',1,...
        'savesolverinput',0,'savesolveroutput',0,'debug',1)...
    );

%fileID = fopen(strcat('result_matlab','_inf',string(inf_level),'_n',string(n_out),'_',distLabel,'.txt'),'w');
lambdavalue = value(lambda);
fprintf('lambda = %.16g\n', lambdavalue);
%fprintf(fileID,'lambda = %lf\n', lambdavalue);
%fclose(fileID);

if FlagCalculateDuals
	X = dual(constraint);
	save dualconstraint X
end

