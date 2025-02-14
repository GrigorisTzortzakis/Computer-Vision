% Load video
vid = VideoReader('video1_high.avi');
% Get frames with larger separation
frame1 = readFrame(vid); % Template
frame20 = readFrame(vid); % Skip to frame 20
for i=1:18
    frame20 = readFrame(vid);
end

% Modify both frames' brightness and contrast
frame1 = 2.5 * frame1 + 40; % Increase template brightness and contrast
frame20 = 2.9 * frame20 + 60; % Increase image brightness and contrast

% Set parameters
levels = 1; % Number of pyramid levels
noi = 50; % Number of iterations
transform = 'affine';
% Initialize with identity transform
delta_p_init = [1 0 0; 0 1 0];
% Call ECC algorithm with frame 1 and 20
[results, results_lk, MSE, rho, MSELK] = ecc_lk_alignment(frame20, frame1, levels, noi, transform, delta_p_init);