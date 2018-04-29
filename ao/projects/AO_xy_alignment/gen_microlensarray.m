clc;clear;close all;
% This script generates a microlens array image on the SLM for calibration

% set size of the SLM
fileID = fopen('../../temp/SLM_sizes.txt');
N = fliplr(cell2mat(textscan(fileID, '%d')).');
fclose(fileID);
N = double(N);
K = [9 16]; % for holoeye

% set number of lenslets
M = [9 10]; % for Holoeye
pad_size = [0 0];

% set lenslet properties
scale = 8.335e-4;
% scale = 8.335e-4 / 2;

% get single lenslet
dim = round(N./K) - 2*pad_size;
[y,x] = meshgrid(1:dim(1),1:dim(2));
x = x - mean2(x);
y = y - mean2(y);
fun = -scale * (x.^2 + y.^2);
fun = padarray(fun, pad_size);

% get the centers
base = (K - M)/2 .* dim + dim/2 + 0.5;
x = base(1):dim(1):(M(1)-1)*dim(1)+base(1);
y = base(2):dim(2):(M(2)-1)*dim(2)+base(2);

% generate the whole pattern, and wrap to [0 255] as [0 2*pi]
phase = repmat(fun, M);
phase = phase - mean2(phase);
phase = padarray(phase, (N-size(phase))/2);
phase = wrapTo2Pi(phase*2*pi)/(2*pi);
phase_mod = uint8(round(phase*255));

% show it
figure;     imshow(phase_mod,[]);       axis on;        hold on;
arrayfun(@(j) arrayfun(@(i) plot(y(i),x(j),'bo'),1:length(y),'un',0),1:length(x),'un',0);
pause(3);

% save it
save('microlens_cali','phase','phase_mod','M','pad_size','scale','x','y');
imwrite(phase_mod, 'cali_microlens_img.png');
