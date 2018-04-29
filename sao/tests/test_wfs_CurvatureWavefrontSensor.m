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
z = 100e-6;                         % distance between mask and sensor
lambda = 550e-9;                    % wavelenth

% get parameters of the scalar field
Udim = factor*pdim;
sampling_rate = pixel_size/factor;

% simulate wavefront
wavefront_range = 10;

% check your parameters
sdoss_para_check(pdim, pixel_size, factor, lambda, z, 'AS');

% set SNR
SNR = 30;

% generate sample scalar field (pre-defined)
[U0_AS, phi_GT] = sdoss_gen_scalarU(Udim, wavefront_range, 'gaussian', 'AS');
[I1, I0, I2] = sao_wfs_CurvatureWavefrontSensor(U0_AS, pdim, sampling_rate, lambda, z, SNR);

% show sensor image
figure;     imshow([I1 I0 I2],[]);
title(['captured images, wavefront range = ' num2str(wavefront_range) '\lambda']);

% run TIE solver
phi_tie = tie_ours(I1, I0, I2, lambda, z);

% normarlize
phi_tie = phi_tie - min(phi_tie(:));
phi_GT  = imresize(phi_GT, 1/factor);
phi_GT  = phi_GT - min(phi_GT(:));

% show results
map = coolwarm(256);
mydisplay(phi_GT, wavefront_range, map);    title('ground truth');
mydisplay(phi_tie,wavefront_range, map);    title('TIE');
