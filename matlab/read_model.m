% reading saved data from a matfile
% input:
%   - a file path pointing to score.mat
% outputs:
%   - bar_C_K: the running average of the truncated condition number of FIM 
%   - L_K: the weighted cumulative sum of the energy of FIM
%   - x_epoch: epoch scaled x-axis values
%   - x_iter: iteration scaled x-axis values
%   - x_iterlr: sum of learning rates per iteration scaled x-axis values
%   - top1: #epoch by 2 matrix, the first column is the training error, the
%           second colum is the testing error
function  [bar_C_K, L_K, x_epoch, x_iter, x_iterlr, top1] = read_model(fpath)

load(fpath)

c_k = [];
l_k = [];

x_epoch = []; 
x_iter = []; 
x_iterlr = []; 

for e=epoch'
  data_e = trainCond.(['e',num2str(e)]);
  data_e = double(data_e);
  sampled_iterations = find(sum(data_e, 2) ~= 0);
  sampling_ratio = size(sampled_iterations,1)/size(data_e,1);
  data_e = data_e(sampled_iterations,:);
  
  lr_e = data_e(:,2);
  batch_size_e = data_e(:,3);
  eigens_e = data_e(:,4:end-1);
  max_eigen_e = max(eigens_e,[], 2);
  eigens_e(eigens_e==0) = Inf;
  min_eigen_e = min(eigens_e, [], 2);
  
  eigens_e(eigens_e==Inf) = 0;
  
  
  c_k = cat(1, c_k, max_eigen_e./min_eigen_e);
  l_k = cat(1, l_k,lr_e./batch_size_e.*sqrt(sum(eigens_e.^2,2)));
  
  x_iter = cat(1, x_iter, ones(size(lr_e)));
  x_iterlr = cat(1, x_iterlr, lr_e);
  x_epoch = cat(1, x_epoch, e-1+cumsum(ones(size(lr_e)))/size(data_e,1));
end

x_iter = cumsum(x_iter);
x_iterlr = cumsum(x_iterlr);
bar_C_K = cumsum(c_k)./x_iter;
L_K = cumsum(l_k);

L_K = L_K / sampling_ratio; % sampling affects L_K but not \bar C_K