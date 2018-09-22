% An example of visualizing all the models in a directory
function plot_multiple_models
global pLines;
pLines = [];

dpath = '../data/cifar10_saved_data';
d = dir(dpath);
d = d(3:end-1);

colors = [    0    0.4470    0.7410
  0.8500    0.3250    0.0980
  0.9290    0.6940    0.1250
  0.4940    0.1840    0.5560
  0.4660    0.6740    0.1880
  0.3010    0.7450    0.9330
  0.6350    0.0780    0.1840];

lr_colormap = [0.4,  0.2,  0.1,  0.05, 0.025;
  2,    4,    3,     1,     5];


% read data from all sub-folders
points = {};
for i=1:size(d,1)
  fpath = fullfile(dpath, d(i).name, '1/score.mat');
  [bar_C_K, L_K, x_epoch, x_iter, x_iterlr, err] = read_model(fpath);
  
  strparts = strsplit(d(i).name, '_');
  
  if size(strparts, 2) == 3
    lr = strparts{3};
    lr = str2num(lr(3:end));
    
    sz = strparts{2};
    sz = str2num(sz(5:end));
    
    p = struct();
    p.dim1 = bar_C_K;
    p.dim1Name = '\\bar C_K';
    p.dim2 = L_K;
    p.dim2Name = 'L_K';
    p.dim3 = err;
    p.dataName = sprintf(['size', num2str(sz), '-lr', num2str(lr)]);
    p.lr = lr;
    p.sz = sz;
    points{end+1} = p;
  end
end



figure(1)
clf;
hold on

background(points, {'size8_lr0.4'}) % excludes the model densenet_size8_lr0.4 because it did not converge
offset = get(gca, 'Zlim');
offset = offset(1);

%% lines
names = {...'size8-lr0.4', ...
  'size16-lr0.4', 'size32-lr0.4', ...
  'size64-lr0.4', 'size128-lr0.4', 'size256-lr0.4'};
plotLines(points, names, colors(lr_colormap(2,lr_colormap(1,:)==0.4),:))

names = {'size8-lr0.2', 'size16-lr0.2', 'size32-lr0.2', ...
  'size64-lr0.2', 'size128-lr0.2', 'size256-lr0.2'};
plotLines(points, names, colors(lr_colormap(2,lr_colormap(1,:)==0.2),:))

names = {'size8-lr0.1', 'size16-lr0.1', 'size32-lr0.1', ...
  'size64-lr0.1', 'size128-lr0.1', ...
  'size256-lr0.1'};
plotLines(points, names, colors(lr_colormap(2,lr_colormap(1,:)==0.1),:))

names = {'size8-lr0.05', 'size16-lr0.05', 'size32-lr0.05', ...
  'size64-lr0.05', 'size128-lr0.05', 'size256-lr0.05'};
plotLines(points, names, colors(lr_colormap(2,lr_colormap(1,:)==0.05),:))

names = {'size8-lr0.025', 'size16-lr0.025', 'size32-lr0.025', ...
  'size64-lr0.025', 'size128-lr0.025', 'size256-lr0.025'};
plotLines(points, names, colors(lr_colormap(2,lr_colormap(1,:)==0.025),:))

for i = 1:size(points,2)
  p = points{i};
  
  x = double(p.dim1(end));
  y = double(p.dim2(end));
  z = p.dim3(end, 2); % 1 for training; 2 for testing error
  
  dataColor = colors(lr_colormap(2,lr_colormap(1,:)==p.lr),:);
  plot3([x x],[y y],[0 z], 'Color', dataColor, 'LineStyle', '--');
  scatter3(x, y, z, 40, dataColor, 'filled', 'MarkerEdgeColor', [0 0 0])
  scatter3(x, y, 0+offset, 40, dataColor, 'filled', 'MarkerEdgeColor', [0 0 0])
  text(x, y, z, sprintf('s%d', p.sz), ...
    'Interpreter', 'none', ...
    'FontSize', 8, 'Color', [0 0 0], 'FontWeight', 'bold');
end

set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
grid on
view([-45,45])
set(gcf, 'Position', [500 500 1024 768])

xlabel(p.dim1Name, 'FontSize', 14, 'FontWeight', 'bold')
ylabel(p.dim2Name, 'FontSize', 14, 'FontWeight', 'bold')
zlabel('Testing Error', 'FontSize', 14, 'FontWeight', 'bold')

legend(pLines, {'lr0.4', 'lr0.2', 'lr0.1', 'lr0.05', 'lr0.025'}, 'location', 'North')
title('DenseNet')
% field interpolation
function background(points, exclusion_list)


numpoints = size(points,2);
matrix = zeros(numpoints, 2);
err = ones(numpoints,1)*-1;
for i=1:numpoints
  p = points{i};
  if any(cellfun(@(x) strcmp(x, p.dataName), exclusion_list))
    continue;
  end
  fprintf('[interpolation] contour map includes: %s\n', p.dataName)
  matrix(i,:) = [p.dim1(end) p.dim2(end)];
  err(i) = p.dim3(end, 2); % 1 for training; 2 for testing
end

matrix = matrix(sum(matrix,2)~=0,:);

err = err(err~=-1);

matrix(:,1) = log(matrix(:,1));
d1max = max(matrix(:,1));
d1min = min(matrix(:,1));
matrix(:,1) = (matrix(:,1)-d1min)/(d1max-d1min);

matrix(:,2) = log(matrix(:,2));
d2max = max(matrix(:,2));
d2min = min(matrix(:,2));
matrix(:,2) = (matrix(:,2)-d2min)/(d2max-d2min);

X = exp(d1min:(d1max-d1min)/50:d1max);
Y = exp(d2min:(d2max-d2min)/50:d2max);
[X,Y]=meshgrid(X,Y);
x = (log(X(:))-d1min)/(d1max-d1min);
y = (log(Y(:))-d2min)/(d2max-d2min);

nummeshpoints=size(x,1);
errEstimated = zeros(nummeshpoints,1);
% knn 5
[IDX,D] = knnsearch(matrix(:,1:2), [x, y],'K', 5);
for i=1:nummeshpoints
  %  i
  d = 1./(D(i,:));
  s = (d+1)./(1+sum(d));
  errEstimated(i) = sum(err(IDX(i,:))' .* s);
end

errEstimated = reshape(errEstimated, size(X));

[~, h_c] = contour(gca,X,Y,errEstimated, 30 );
zMinVal = min(errEstimated(:))*0.95;
zMaxVal = max(errEstimated(:));
h_c.ContourZLevel = zMinVal;
xlim([min(X(1,:)), max(X(1,:))])
ylim([min(Y(:,1)), max(Y(:,1))])

zlim([zMinVal, zMaxVal])

function plotLines(points, names,color)
global pLines;
xyzs = getPointsByNames(points,names);
if nargin == 2
  h=plot3(xyzs(:,1),xyzs(:,2),xyzs(:,3), 'linewidth', 2);
else
  h=plot3(xyzs(:,1),xyzs(:,2),xyzs(:,3), 'linewidth', 2, 'Color', color);
end

pLines = [pLines h];

function xyzs = getPointsByNames(points, names)
xyzs = [];
for i = 1:size(names,2)
  fprintf(names{i})
  xyzs = [xyzs; findPointByName(points, names{i})];
  fprintf(': %0.2f\n',xyzs(end,3));
end

function xyz = findPointByName(points, name)

for i=1:size(points,2)
  p = points{i};
  if strcmp(p.dataName, name)
    xyz = [p.dim1(end), p.dim2(end), p.dim3(end,2)];
    break;
  end
end

if ~exist('xyz', 'var')
  fprintf('can not find data with name: %s\n', name);
end