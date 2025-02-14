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

% Storage for all a values
a_values = [power(6,1/3) power(12,1/3) power(18,1/3)];  % values in gray levels
all_MSE = zeros(length(a_values), 100, noi);
all_MSELK = zeros(length(a_values), 100, noi);
all_rho = zeros(length(a_values), 100, noi);

% Convert to double
frame1 = double(frame1);
frame20 = double(frame20);

% Test all a values
for a_idx = 1:length(a_values)
    a = a_values(a_idx);
    fprintf('\nTesting a = %.4f\n', a);
    
    % Run 100 experiments
    for exp = 1:100
        % Add uniform noise U[-a,a] directly in gray levels
        noisy_frame1 = frame1 + (2*a)*rand(size(frame1)) - a;
        noisy_frame20 = frame20 + (2*a)*rand(size(frame20)) - a;
        
        % Ensure valid range [0,255]
        noisy_frame1 = max(0, min(255, noisy_frame1));
        noisy_frame20 = max(0, min(255, noisy_frame20));
        
        % Run alignment
        [~, ~, MSE, rho, MSELK] = ecc_lk_alignment(uint8(noisy_frame20), ...
            uint8(noisy_frame1), levels, noi, transform, delta_p_init);
            
        all_MSE(a_idx, exp, :) = MSE;
        all_MSELK(a_idx, exp, :) = MSELK;
        all_rho(a_idx, exp, :) = rho;
    end
end

% Plot results
figure('Position', [100 100 900 600]);
subplot(2,1,1);
colors = {'b', 'r', 'k'};
for a_idx = 1:length(a_values)
    mse_mean = squeeze(mean(all_MSE(a_idx,:,:), 2));
    mselk_mean = squeeze(mean(all_MSELK(a_idx,:,:), 2));
    
    plot(mse_mean, [colors{a_idx} '-'], 'DisplayName', sprintf('ECC a=%.2f', a_values(a_idx)));
    hold on;
    plot(mselk_mean, [colors{a_idx} '--'], 'DisplayName', sprintf('LK a=%.2f', a_values(a_idx)));
end
title('Meso MSE gia ola ta α');
legend;
grid on;

subplot(2,1,2);
for a_idx = 1:length(a_values)
    rho_mean = squeeze(mean(all_rho(a_idx,:,:), 2));
    plot(rho_mean, colors{a_idx}, 'DisplayName', sprintf('a=%.2f', a_values(a_idx)));
    hold on;
end
title('Autosisxetish');
legend;
grid on;