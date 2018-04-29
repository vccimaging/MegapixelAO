function tA = sdoss_gen_mask(pdim, factor, k, n, a, method)

switch k
    case 'rand'
        rng(2015)
        tB = rand(pdim);
        tB = floor(tB * 2^n)/(2^n-1);
    case 'rand_phase'
        rng(2015)
        tB = double(uint8(255*rand(pdim)))/255;
    case 'ones'
        tB = ones(pdim);
    case 'img'
        tB = double(imresize(imread('boat.png'), pdim));
    case 'wavelet'
        if pdim(1) ~= pdim(2)
            error('waveletnoise mask only supports square')
        end
        temp = 1:ceil(log(min(pdim))/log(2))-1;
        J = min(pdim)*ones(numel(temp),1)./2.^temp';
        tB = gen_waveletnoise(min(pdim), J);
    case 'binary_phase'
        rng(2015)
        pos = randperm(prod(pdim),round(prod(pdim)/2));
        tB = ones(pdim);
        tB(pos) = -1;
    case 'binary'
        rng(2015)
        pos = randperm(prod(pdim),round(prod(pdim)/2));
        tB = zeros(pdim);
        tB(pos) = 1;
    case 'circular'
        tB = zeros(pdim);
        N = pdim(1);
        M = pdim(2);
        v = (1:N) - (N+1)/2;
        u = (1:M) - (M+1)/2;
        [x,y] = meshgrid(v,u);
        tB(sqrt(x.^2 + y.^2) <= a) = 1;
    case 'square'
        tB = zeros(pdim);
        N = pdim(1);
        M = pdim(2);
        v = (1:N) - (N+1)/2;
        u = (1:M) - (M+1)/2;
        [x,y] = meshgrid(v,u);
        tB(logical((abs(x) <= a/2) .* (abs(y) <= a/2))) = 1;
    otherwise
        error('undefined mask type')
end

tA = zeros(pdim*factor);
for i = 1:factor
    for j = 1:factor
        tA(i:factor:end,j:factor:end) = tB;
    end
end

switch method
    case 'DI'
        return
    case 'AS'
        tA = padarray(tA, pdim*factor);
    otherwise
        error('Undefined method. Method can only be either DI or AS');
end

    function mask = gen_waveletnoise(k,J)
        
        rng(1992);
        mask = zeros(k,k);
        
        % make sure J is an integeral number
        J = floor(J);
        J(J == 1) = [];
        tempn = numel(J);
        
        for ii = 1:tempn
            temp_m = rand(J(ii),J(ii));
            temp_m = temp_m - imresize(imresize(temp_m, 0.5), 2);
            mask = mask + imresize(temp_m, k/J(ii));
        end
        mask = mask - min(mask(:));
        mask = mask / max(mask(:));
        
    end

end
