function varargout = splitData(data, perc, seed)
% Randomly splits data into length(perc) subsets where the ith element in 
% perc is the percentage of data that should belong to subset i. 
% Splits over columns if perc is row vector. Splits over rows if perc is 
% a column vector. The sum of perc must be equal to 1.
% Seed sets the seed for the randomized split. [defaul=0] 
%
% EXAMPLE: Split columns of data into 80% traindata, 10% validation data and 10% testdata
%
%           [traindata, valdata, testdata] = split(data, [0.8 0.1 0.1])
%
% EXAMPLE: Split rows of data into 80% traindata, 10% validation data and 10% testdata
%
%           [traindata, valdata, testdata] = split(data, [0.8; 0.1; 0.1])
%
% EXAMPLE: If you only want 2 subsets you can make them like this:
%
%           [traindata, testdata] = split(data, [0.7; 0.3])
%

if abs(sum(perc)-1) > 1e-10
    error('Sum of perc is not 1');
end

%orientation = argmax(size(perc));
[~, orientation] = max(size(perc));

if nargin<3
    rand('state', 0)
else
    rand('state', seed)
end
S=length(perc); %number of subsets
N=size(data, orientation);
k=randperm(N);

perc = [0; perc(:)];

for i=1:S
    if orientation==1
        varargout{i} = data(k(1+floor(N*sum(perc(1:i))):floor(N*sum(perc(1:i+1)))),:);
    else
        varargout{i} = data(:,k(1+floor(N*sum(perc(1:i))):floor(N*sum(perc(1:i+1)))));
    end
end

end