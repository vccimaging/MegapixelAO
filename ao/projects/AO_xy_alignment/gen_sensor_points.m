clc;clear;close all;

% add path
addpath('../../../solvers');   

% read and average the images
delay_frame = 10;
temp = dir('./data');
img = arrayfun(@(i) imread(['./data/' temp(i).name]), 3:length(temp),'un',0);
img = reshape(cell2mat(img), [size(img{1}) (length(temp)-2)]);
img_ref = mean(img(:,:,1:size(img,3)/2), 3);
img_cap = mean(img(:,:,1+size(img,3)/2+delay_frame:end), 3);
mean2(var(double(img(:,:,1:size(img,3)/2)), 0, 3))
mean2(var(double(img(:,:,1+size(img,3)/2+delay_frame:end)), 0, 3))
clear temp img;

% convert to grayscale if needed
if (size(img_ref,3) == 3)
    img_ref = rgb2gray(img_ref);
    img_cap = rgb2gray(img_cap);
end

% save images
save('imgs','img_ref','img_cap');


%%

% load previously saved SLM calibration pattern data
load('microlens_cali.mat');

% crop image
crop_size = [1024 1024];
img = cat(3, img_ref, img_cap);
L = (size(img_ref) - crop_size)/2;

% set algorithm parameters
beta = 10;
opt.mu = [2e1 1e2];
opt.L = {[16 16],[8 8]};
opt.iter = 10;
opt.isverbose = 1;
warping_iter = [3 3];

% the algorithm
phi = main_DCT_pyramid_warping(img, beta, opt, warping_iter);
phi = 0.05 * -phi;
figure;     mesh(phi);      axis tight;

% detect peak points
radii = [20, 55];       % you may need to tune it by hand
[c,~] = imfindcircles(phi,radii);

% check if the number of detected points are correct with your loaded
% microlens number
if numel(c) ~= 2*prod(M)
    error('Incorrect detected points; Please re-tune "radii" and re-run this .m code!');
end

% show the reconstructed phase and the detected points
figure;     imshow(phi,[]);     axis ij; axis on;hold on
plot(c(:,1),c(:,2),'r+')

% get the mean x and y coordinates
x_coord = mean(reshape(sort(c(:,1)),M));
y_coord = mean(reshape(sort(c(:,2)),fliplr(M)));

% get a data matrix and fillin each detected circle center
mat_sensor = zeros([M 2]);
eps = 50;           % you may need to tune it by hand
for i = 1:prod(M)
    x_temp = find(abs(c(i,1) - x_coord) < eps);
    y_temp = find(abs(c(i,2) - y_coord) < eps);
    mat_sensor(y_temp, x_temp, :) = c(i,:);
end

% check if the sorted points are correct
if (prod(mat_sensor(:)) == 0)
    error('Incorrect sorted points; Pleas tune "eps" and re-run this .m code!');
end

% re-formulate the SLM data into the same format
mat_SLM = flip(cat(3, repmat(x', [1 M(2)]), repmat(y, [M(1) 1])),3);

% get the points
y_i = mat_sensor(:,:,1);    y_i = y_i(:);
x_i = mat_sensor(:,:,2);    x_i = x_i(:);
y_o = mat_SLM(:,:,1);       y_o = y_o(:);
x_o = mat_SLM(:,:,2);       x_o = x_o(:);

% calculate perspective matrix T
tform = estimateGeometricTransform([y_i-L(2) x_i-L(1)],[y_o x_o],'projective');
T = tform.T;

% write transform matrix T to .txt
fileID = fopen('../../temp/perspective_T.txt','w');
fprintf(fileID,'%d\n', T(:));
fclose(fileID);
