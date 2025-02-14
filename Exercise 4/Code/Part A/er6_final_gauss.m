% Load video
vid = VideoReader('video1_high.avi');
frame1 = readFrame(vid); 
frame20 = readFrame(vid);
for i=1:18
    frame20 = readFrame(vid);
end

% Convert to grayscale if needed
if size(frame1,3) > 1
    frame1 = rgb2gray(frame1);
    frame20 = rgb2gray(frame20);
end

% Parameters
levels = 1;
noi = 15;
transform = 'affine';
delta_p_init = [1 0 0; 0 1 0];

% Storage for all variances
vars = [4 8 12];  % σ² values in gray levels
all_MSE = zeros(length(vars), 100, noi);
all_MSELK = zeros(length(vars), 100, noi);
all_rho = zeros(length(vars), 100, noi);

% Convert to double
frame1 = double(frame1);
frame20 = double(frame20);

% Test all variances
for var_idx = 1:length(vars)
    var = vars(var_idx);
    fprintf('\nTesting variance = %d\n', var);
    
    % Run 100 experiments
    for exp = 1:100
        % Add Gaussian noise N(0,σ²) directly in gray levels
        noisy_frame1 = frame1 + randn(size(frame1))*sqrt(var);
        noisy_frame20 = frame20 + randn(size(frame20))*sqrt(var);
        
        % Ensure valid range [0,255]
        noisy_frame1 = max(0, min(255, noisy_frame1));
        noisy_frame20 = max(0, min(255, noisy_frame20));
        
        % Run alignment
        [~, ~, MSE, rho, MSELK] = ecc_lk_alignment(uint8(noisy_frame20), ...
            uint8(noisy_frame1), levels, noi, transform, delta_p_init);
            
        all_MSE(var_idx, exp, :) = MSE;
        all_MSELK(var_idx, exp, :) = MSELK;
        all_rho(var_idx, exp, :) = rho;
    end
end

% Plot results
figure('Position', [100 100 900 600]);
subplot(2,1,1);
colors = {'b', 'r', 'k'};
for var_idx = 1:length(vars)
    mse_mean = squeeze(mean(all_MSE(var_idx,:,:), 2));
    mselk_mean = squeeze(mean(all_MSELK(var_idx,:,:), 2));
    
    plot(mse_mean, [colors{var_idx} '-'], 'DisplayName', sprintf('ECC σ^2=%d', vars(var_idx)));
    hold on;
    plot(mselk_mean, [colors{var_idx} '--'], 'DisplayName', sprintf('LK σ^2=%d', vars(var_idx)));
end
title('Meso MSE gia ola ta σ');
legend;
grid on;

subplot(2,1,2);
for var_idx = 1:length(vars)
    rho_mean = squeeze(mean(all_rho(var_idx,:,:), 2));
    plot(rho_mean, colors{var_idx}, 'DisplayName', sprintf('σ^2=%d', vars(var_idx)));
    hold on;
end
title('Autosisxetish');
legend;
grid on;