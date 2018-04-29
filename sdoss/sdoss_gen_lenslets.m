function lens_tran = sdoss_gen_lenslets(Udim, D, lambda, f)

[y, x] = meshgrid(linspace(-D(1)/2, D(1)/2, Udim(1)), ...
                  linspace(-D(2)/2, D(2)/2, Udim(2)));

lens_tran = exp(-1i*(2*pi/lambda)*(x.^2 + y.^2)/(2*f));

