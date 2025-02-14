% Load video
vid = VideoReader('video1_high.avi');

% Get frame 1
frame1 = readFrame(vid); % Template

% Skip to frame 60 by reading 58 more frames
frame60 = readFrame(vid); % This gets frame 2
for i=1:58
    frame60 = readFrame(vid);
end

% Set parameters
levels = 1; % Number of pyramid levels
noi = 50; % Number of iterations
transform = 'affine';

% Initialize with identity transform
delta_p_init = [1 0 0; 0 1 0];

% Call ECC algorithm with frame 1 and 60
[results, results_lk, MSE, rho, MSELK] = ecc_lk_alignment(frame60, frame1, levels, noi, transform, delta_p_init);