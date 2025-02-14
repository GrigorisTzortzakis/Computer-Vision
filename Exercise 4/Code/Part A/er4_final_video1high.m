

% 1) Load the video
vid = VideoReader('video1_low.avi');

% 2) Read frame #1 (template)
frame1 = readFrame(vid);

% 3) Read frame #2 (moving image)
frame2 = readFrame(vid);

% 4) Set parameters
levels       = 1;             % Number of pyramid levels
noi          = 15;            % Number of iterations
transform    = 'affine';      % Warp model
delta_p_init = [1 0 0; ...    % Identity transform (2x3) for affine
                0 1 0];

% 5) Call ECC/LK alignment
[results, results_lk, MSE, rho, MSELK] = ecc_lk_alignment( ...
    frame2, ...    % moving image
    frame1, ...    % template
    levels, ...    % 1
    noi, ...       % 15
    transform, ... % 'affine'
    delta_p_init); % identity initialization

% 6) Display final ECC and LK warps
% (Since levels=1 and noi=15, the final warp is at results(1,15).warp)
disp('Final ECC warp (2x3):');
disp(results(1,15).warp);

disp('Final LK warp (2x3):');
disp(results_lk(1,15).warp);

% (Optional) Examine MSE, MSELK, rho, etc.
disp('MSE (ECC) across iterations:');
disp(MSE);

disp('MSE (LK) across iterations:');
disp(MSELK);

disp('ECC correlation (rho) across iterations:');
disp(rho);
