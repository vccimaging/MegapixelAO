clc;clear;close all;
addpath(genpath('../../../utilities/'));

% datapath
datapath = './data/';

% read image
img_ref = double(imread([datapath 'img_reference.png']));
img_cap = double(imread([datapath 'img_capture.png']));

% set active area
M = [992 992];
N = [1024 1024];

% crop size
S = (size(img_ref)-M)/2;

% crop data
img_ref = img_ref(1+S(1):end-S(1),1+S(2):end-S(2));
img_cap = img_cap(1+S(1):end-S(1),1+S(2):end-S(2));

% the input data
img = cat(3, img_ref, img_cap);

% set parameters
theta = 1/8;
iter  = 10;
alp   = 0.95;

% ground truth
img_rof_cpu = fista_rof(img, theta, iter, alp);
img_rof_cpu = img_rof_cpu(:,:,1);

% load GPU result
test = readFlowFile('test_fista_rof.flo');
img_rof_gpu = test(:,:,1);clear test;

% show results
figure;     imshow([img_rof_cpu img_rof_gpu img_rof_cpu - img_rof_gpu],[]);
title('CPU / GPU / Difference');
disp(['max diff = ' num2str(norm(img_rof_gpu(:) - img_rof_cpu(:),'inf'))]);

