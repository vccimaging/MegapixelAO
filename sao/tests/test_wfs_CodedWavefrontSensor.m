clc;clear;close all;
%%%% Warning: Please run it on a large memory PC! 

% check memory
mem = memory;
if (mem.MemAvailableAllArrays / 1e9) < 8
    error('Memory < 8 GB; Please run this code on a larger memory PC!')
end

% add path
addpath('../')
addpath('../../sdoss');
addpath('../../solvers/');
addpath('../../utilities/');

% to be reproducible
rng('default')

% set parameters
pdim = [512 512];                   % sensor spatial resolution
pixel_size = 6.45e-6;               % sensor pixel size
factor = 5;                         % upsampling factor
z = 1.5e-3;                         % distance between mask and sensor
lambda = 550e-9;                    % wavelenth
alpha = pixel_size^2/(lambda*z);    % scale factor

% get parameters of the scalar field
Udim = factor*pdim;
sampling_rate = pixel_size/factor;

% set noise
SNR = 30;                           % signal-to-noise ratio [dB]

% generate a binary mask
tA_AS = sdoss_gen_mask(pdim, factor, 'rand', 1, [], 'AS');
mask_AS = sdoss_getmask(tA_AS, factor, 'AS');

% SDOSS AS imaging
sdoss_para_check(pdim, pixel_size, factor, lambda, z, 'AS');

% calibration of Coded Wavefront Sensor
img_reference = sao_wfs_CodedWavefrontSensor(padarray(ones(Udim),Udim), tA_AS, ...
                                    pdim, sampling_rate, lambda, z, SNR);

% set wavefront range
wavefront_range = 10;

% run the simulation
[U0_AS, phi_GT] = sdoss_gen_scalarU(Udim, wavefront_range, 'gaussian', 'AS');
img_capture = sao_wfs_CodedWavefrontSensor(U0_AS, tA_AS, pdim, sampling_rate, lambda, z, SNR);

% show images
figure;     imshow(img_reference, []);    title('image\_reference');
figure;     imshow(img_capture, []);      title('image\_captured');
figure;     imshow(img_capture - img_reference,[]);
title('their difference');

% set algorithm parameters
beta = 1;
opt.mu = [100 100];
opt.iter = 10;
opt.isverbose = 0;
opt.L = {[2 2],[2 2]};
warping_iter = [2 3];

% call the wavefront solver
phi_ours = alpha * main_wavefront_solver(cat(3, img_reference, img_capture), ...
                        beta, opt, warping_iter);

% normarlize
phi_ours = phi_ours - min(phi_ours(:));
phi_GT = imresize(phi_GT, 1/factor);
phi_GT = phi_GT - min(phi_GT(:));

% show results
map = coolwarm(256);
mydisplay(phi_GT,  wavefront_range, map);    title('groud truth');
mydisplay(phi_ours,wavefront_range, map);    title('ours');

