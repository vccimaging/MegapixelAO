function newU = sdoss_propagation(U0, lambda, sampling_rate, z, method)

% use GPU if Nvidia GPU card is installed and recognizable by MATLAB
% try 
%     t1 = gpuDevice;
%     t2 = whos('U0');
%     if t2.bytes  <  1/100 * t1.AvailableMemory
% %         disp(num2str(t2.bytes / t1.AvailableMemory))
%         isgpu = 1;
%         U0 = gpuArray(U0);
%         disp('SDOSS is using GPU')
%     else
%         isgpu = 0;
%         disp('SDOSS is using CPU; GPU memory insufficient')
%     end
% catch
%     isgpu = 0;
%     disp('SDOSS is using CPU; GPU unavailable')
% end

% With the Simpsion's weighted matrix, we found the precision decreases.
% Hence, we do not use the weighted matrix, which was used in the paper.
% Decomment following codes if you want a Simpsion's weighted matrix.
switch 'nonsimpson'
    case 'simpson'
        B = gen_simpson_weimat(size(U0,1));
        W = B'*B;
        U0 = W.*U0;
    case 'nonsimpson'
end

switch method
    case 'DI'
%         try
            newU = FFT_DI_propagation(U0, lambda, sampling_rate, z);
%         catch
%             newU = FFT_DI_propagation(gather(U0), lambda, sampling_rate, z);
%         end
    case 'AS'
%         try
            newU = diff_propagation(U0, lambda, sampling_rate, z);
%         catch
%             newU = diff_propagation(gather(U0), lambda, sampling_rate, z);
%         end
    case 'AS_periodic'
            newU = diff_propagation(U0, lambda, sampling_rate, z);
    otherwise
        error('Undefined method. Method can only be either DI or AS');
end

% if isgpu
%     newU = gather(newU);
% end

    function newU = FFT_DI_propagation(U0, lambda, sampling_rate, z)
        
        k = 2*pi/lambda;
        size_U0 = size(U0);
        
        x = ((0:size_U0(1)-1)-floor(size_U0(1)/2)) * sampling_rate;
        y = ((0:size_U0(2)-1)-floor(size_U0(2)/2)) * sampling_rate;
        
        switch 'DI'
            case 'DI'
                g = @DI;
            case 'fresnel'
                g = @fresnel;
        end
        
        switch 'fft'
            case 'conv'
                newU = conv2(U0, g(x,y,z), 'same');
            case 'fft'
                g_xyz = g(x,y,z);
                dim = size_U0 + size(g_xyz) - 1;
                newU = ifft2(fft2(U0,dim(1),dim(2)) .* ...
                        fft2(g_xyz,dim(1),dim(2)));
                crop_dim = max(size_U0,size(g_xyz));
                crop_offset = floor((size_U0 + size(g_xyz))/4);
                newU = newU(1+crop_offset(1):crop_dim(1)+crop_offset(1),...
                    1+crop_offset(2) : crop_dim(2)+crop_offset(2));
        end
        
        function out = DI(x,y,z)
            [Y, X] = meshgrid(y,x);
            r = sign(z)*sqrt(X.^2 + Y.^2 + z^2);
            out = 1/(2*pi) * z * exp(1i*k*r)./r.^3 .* (1 - 1i*k*r);
            out = out .* sampling_rate^2;
        end
        
        function out = fresnel(x,y,z)
            [Y, X] = meshgrid(y,x);
            out = exp(1i*k*z + 1i*k/(2*z)*(X.^2 + Y.^2)) / (1i*lambda*z);
            out = out .* sampling_rate^2;
        end
        
    end

    function newU = diff_propagation(U, lambda, sampling_rate, z)
        
        f0 = 1/sampling_rate;
        Uspectrum = fftshift(fft2(U));
        
        n1 = (0:size(Uspectrum,1)-1) - floor(size(Uspectrum,1)/2);
        n2 = (0:size(Uspectrum,2)-1) - floor(size(Uspectrum,2)/2);
        
        [beta, alpha] = meshgrid(lambda * n2/size(Uspectrum,2) * f0, ...
            lambda * n1/size(Uspectrum,1) * f0);
        
        newUspectrum = (Uspectrum .* ...
            exp(2i*pi/lambda*sqrt(1 - alpha.^2 - beta.^2)*z));
        
        % lowpass the spectrum to avoid aliasing
        
        % This is refered to paper:
        % Matsushima, Kyoji, and Tomoyoshi Shimobaba. "Band-limited angular
        % spectrum method for numerical simulation of free-space propagation
        % in far and near fields." Optics express 17.22 (2009): 19662-19673.
        delta_uv = 1./(2*size(U).*sampling_rate);
        uv_limit = 1 / lambda ./ sqrt(1 + (2*delta_uv*z).^2);
        if strcmp(method, 'AS')
            newUspectrum = rect(newUspectrum, uv_limit/f0.*size(U));
%       Theoritically it should be the above commented one. However 1/1.1
%       dones better than 1/1. Thus we use 1/1.1 instead.
        end
        newU = ifft2(ifftshift(newUspectrum));
        
        function x = rect(x, a)
            u = (1:size(x,1)) - round(size(x,1)/2);
            v = (1:size(x,2)) - round(size(x,2)/2);
            if max(u)~=-min(u)
                u(u==0) = [];
                u = [-max(u) u];
            end
            if max(v)~=-min(v)
                v(v==0) = [];
                v = [-max(v) v];
            end
            [X,Y] = meshgrid(u,v);
            x(logical((abs(X) > a(1)))) = 0;
            x(logical((abs(Y) > a(2)))) = 0;
        end
        
    end

    function B = gen_simpson_weimat(n)
        
        if mod(n,2)
            B = [1 repmat([4 2], [1 (n-3)/2]) 4 1]/3;
        else
            error('I dont know what to write for even number n!')
        end
        
    end
end
