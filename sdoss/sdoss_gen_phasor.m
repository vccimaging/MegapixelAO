function phasor = sdoss_gen_phasor(Udim, k, varargin)

switch k
    case 'rand'
        phasor = exp(1i*rand(Udim));
    case 'prism'
        lambda = cell2mat(varargin(1));
        alpha = cell2mat(varargin(2));
        [x,y] = meshgrid( ...
            (1:Udim(1)) - round(Udim(1)/2), ...
            (1:Udim(2)) - round(Udim(2)/2));
        phasor = exp(-1i*pi/lambda*alpha*(x + y));
    case 'lens'
        lambda = cell2mat(varargin(1));
        f = cell2mat(varargin(2));
        [x,y] = meshgrid( ...
            (1:Udim(1)) - round(Udim(1)/2), ...
            (1:Udim(2)) - round(Udim(2)/2));
        phasor = exp(-1i*pi/lambda/f*(x.^2 + y.^2));
    case 'cubic phase'
        lambda = cell2mat(varargin(1));
        alpha = cell2mat(varargin(2));
        [x,y] = meshgrid( ...
            (1:Udim(1)) - round(Udim(1)/2), ...
            (1:Udim(2)) - round(Udim(2)/2));
        phasor = exp(-1i*pi/lambda*alpha*(x.^3 + y.^3));
    otherwise
        error('undefined phasor type')
end

switch method
    case 'DI'
        return
    case 'AS'
        phasor = padarray(phasor, Udim);
    otherwise
        error('Undefined method. Method can only be either DI or AS');
end

end
