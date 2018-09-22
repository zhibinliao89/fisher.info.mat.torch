% An example of visualizing the result of a single model
fpath = '../data/cifar10/cbrcbresnet_size64_lr0.1_example/1/score.mat';

[bar_C_K, L_K, x_epoch, x_iter, x_iterlr, err] = read_model(fpath);

x_type = 'lr';
y_type = 'bar_C_K';

switch x_type
  case 'epoch'
    % plot w.r.t. the # of epochs
    x = x_epoch;
    x_label = 'Epoch';
  case 'iter'
    % plot w.r.t. the # of iterations
    x = x_iter;
    x_label = 'Iteration';
  case 'lr'
    % plot w.r.t. the sum of learning rates
    x = x_iterlr;
    x_label = 'Sum of Learning Rates Per Iteration';
end

switch y_type
  case 'L_K'
    y = L_K;
    y_scale = 'log';
    y_label = 'L_K';
  case 'bar_C_K'
    y = bar_C_K;
    y_scale = 'log';
    y_label = '\\bar C_K';
  case 'train_err'
    y = err(:,1);
    x = 1:size(y,1); % must use epoch scale
    y_scale = 'linear';
    x_label = 'Epoch';
    y_label = 'Training Error';
  case 'test_err'
    y = err(:,2);
    x = 1:size(y,1); % must use epoch scale
    y_scale = 'linear';
    x_label = 'Epoch';
    y_label = 'Testing Error';
end

figure(1)
clf
plot(x, y)
xlim([0, x(end)])
set(gca, 'YScale', y_scale)

xlabel(x_label)
ylabel(y_label)

title(fpath)