function [img_SH, x_peak, y_peak] = sao_wfs_shackhartmann(U0, numLens, D, ...
                            lambda, f, pdim_each, factor, SNR)

% generate lenslet function
lens_tran = sdoss_gen_lenslets(round(size(U0)./numLens),[D D],lambda,f);

% convert U0 to cell
U0 = mat2cell(U0, repmat(size(lens_tran,1), [1 numLens(1)]), ...
                  repmat(size(lens_tran,2), [1 numLens(2)]));

% check sampling condition
% sdoss_para_check(pdim_each, D/pdim_each(1), factor, lambda, f, 'DI');

% define repeat function
rep = @(x) repmat({x}, numLens);

%%%%%%%%%% Following code works under proper sampling condition %%%%%%%%%%%
% propagate through sub-apertures
temp1 = cellfun(@times, U0, rep(lens_tran), 'un', 0);
u2 = cellfun(@sdoss_propagation, temp1, rep(lambda), ...
                        rep(D/pdim_each(1)/factor), rep(f), rep('DI'), 'un', 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% get captured Shack-Hartmann sensor image
img_SH = cellfun(@sdoss_sensor, u2, rep(pdim_each), rep('DI'), 'un', 0);

% add Gaussian noise
img_SH = cellfun(@(I,m,v) I + v*randn(size(I)) + m, img_SH, rep(0), ...
                    rep(var(reshape(cell2mat(img_SH),[],1))/10^(SNR/10)), 'un', 0);

% round to positive
img_SH = cell2mat(img_SH);
img_SH(img_SH < 0) = 0;
img_SH = mat2cell(img_SH, repmat(pdim_each(1), [1 numLens(1)]), ...
                          repmat(pdim_each(2), [1 numLens(2)]));

% find peak positions
[y_lens, x_lens] = meshgrid(linspace(-pdim_each(1)/2,pdim_each(1)/2,pdim_each(1)), ...
                            linspace(-pdim_each(2)/2,pdim_each(2)/2,pdim_each(2)));
x_peak = cellfun(@(img) sum(sum(x_lens.*img))/sum(img(:)), img_SH);
y_peak = cellfun(@(img) sum(sum(y_lens.*img))/sum(img(:)), img_SH);

