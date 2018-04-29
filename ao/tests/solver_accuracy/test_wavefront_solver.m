clc;clear;close all;
addpath('../../../');
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

% set algorithm parameters
img = cat(3, img_ref, img_cap);
beta = 3;
opt.mu = 1e2;
opt.L = {[16 16]};
opt.iter = 10;
opt.isverbose = 0;
warping_iter = 1;

% the algo
img_rof = fista_rof(img, 1/8, 20, 0.95);
phi_cpu = main_DCT_pyramid_warping(img_rof, beta, opt, warping_iter);
phi_cpu = -0.05 * phi_cpu;


%%

% load GPU result
test = readFlowFile('test_solver.flo');
phi_gpu = test(:,:,1);clear test;
phi_gpu = phi_gpu(1+opt.L{1}(1):end-opt.L{1}(1), 1+opt.L{1}(2):end-opt.L{1}(2));
phi_gpu = -0.05 * phi_gpu;

% colormap
map = coolwarm(256);

% show RMS
disp(['CPU RMS = ' num2str(sqrt(sum(phi_cpu(:).^2)/numel(phi_cpu)))]);
disp(['GPU RMS = ' num2str(sqrt(sum(phi_gpu(:).^2)/numel(phi_cpu)))]);

% normalize
phi_cpu = phi_cpu - min(phi_cpu(:));
phi_gpu = phi_gpu - min(phi_gpu(:));

% show results
mydisplay(phi_cpu,10,map);     title('cpu');
mydisplay(phi_gpu,10,map);     title('gpu');
mesh(phi_cpu - phi_gpu);axis tight;	colormap(jet);title('difference');
disp(['max diff = ' num2str(max(max(abs(phi_cpu - phi_gpu))))]);


