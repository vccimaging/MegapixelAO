clc;clear;close all;

% add path
addpath('../');

% to be reproducible
rng('default')

% define parameters
r0 = 0.05;       % coherence diameter [m]
delta = 1e-3;    % grid spacing [m]
N = 2048;        % number of grid points
D = N*delta;     % lengths of the phase screen [m]
L0 = 100;        % outer scale [m]
l0 = 1e-3;       % inner scale [m]

% generate a random draw of an atmospheric phase screen
phz = sao_phasescreen(r0, N, delta, L0, l0);

% show result
figure;     imshow(phz, []);    colormap(jet);      colorbar;
title('phase screen / [\lambda]');

disp(['wavefront rms: ' num2str( rms(phz(:) * 500e-9 * 1e6) ) ' um'])

