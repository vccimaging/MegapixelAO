clc;clear;close all;
% this file generates different wavefronts for phase modulator

% define scalar field dimension 1920 x 1080
fileID = fopen('../../temp/SLM_sizes.txt');
N = fliplr(cell2mat(textscan(fileID, '%d')).');
N = double(N);
fclose(fileID);

% Udim = [1920 1080];
Udim = [1920 1920];

% select an option
% opt = 'cubic';
% opt = 'tilt';
opt = 'spherical';
% opt = 'triangular';
% opt = 'sinusoid';
% opt = 'pla';

% generate the wanted wavefront
switch opt
    case 'pla'
        phase = zeros(Udim);
        coeff = 0;
    case 'rand_phase'
        rng(2015);
        phase = rand(Udim) - 0.5;
    case 'rand_amplitude'
        rng(2015);
        U = rand(Udim);
    case 'cubic'
        coeff = 4e-3;
        [x,y] = meshgrid( 1:Udim(1), 1:Udim(2) );
        x = x - mean2(x);
        y = y - mean2(y);
        x = x * coeff;
        y = y * coeff;
        phase = 2 * (x.^3 + y.^3);
    case 'tilt'
        coeff = 5e-2;
        [phase, ~] = meshgrid(coeff*(1:Udim(2)),coeff*(1:Udim(1)));
    case 'spherical'
        coeff = 4e-5;
        [x,y] = meshgrid( (1:Udim(1)) - round(Udim(1)/2), ...
                          (1:Udim(2)) - round(Udim(2)/2) );
        phase = coeff*(x.^2 + y.^2);
    case 'square'
        a = 100;
        phase = padarray(ones(a,a), (Udim-a)/2);
    case 'triangular'
        coeff = 0.01;
        phase = repmat(sawtooth((1:Udim(1))*coeff), [Udim(1), 1]);
    case 'sinusoid'
        coeff = 0.01;
        [x,y] = meshgrid( (1:Udim(1)) - round(Udim(1)/2), ...
                          (1:Udim(2)) - round(Udim(2)/2) );
        phase = pi * sin( coeff * y);
        phase = phase - min(phase(:));
end
phase = phase';

% crop phase
b = (1920 - N(1))/2;
phase = phase(1+b:1920-b, :);

% normalize to zero-mean
phase = phase - mean2(phase);

% discretize
phase_mod = wrapTo2Pi(phase*2*pi)/(2*pi)*255;

% show results
figure;     imshow(phase_mod, []);      colorbar;     title('SLM phase');

% write results
imwrite(uint8(phase_mod), ['SLM_gt_' opt '_' num2str(coeff) '.png']);

