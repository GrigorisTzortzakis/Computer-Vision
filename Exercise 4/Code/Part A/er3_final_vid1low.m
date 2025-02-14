% Load video
vid = VideoReader('video1_low.avi');
% Get frames with larger separation
frame1 = readFrame(vid); % Template
frame20 = readFrame(vid); % Skip to frame 20
for i=1:18
 frame20 = readFrame(vid);
end
% Set parameters
levels = 1; % Number of pyramid levels
noi = 15; % Number of iterations
transform = 'affine';
% Initialize with identity transform
delta_p_init = [1 0 0; 0 1 0];
% Call ECC algorithm with frame 1 and 20
[results, results_lk, MSE, rho, MSELK] = ecc_lk_alignment(frame20, frame1, levels, noi, transform, delta_p_init);