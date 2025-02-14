

% 1) Load the video
vid = VideoReader('video1_high.avi');

% 2) Read frame #1 (template)
frame1 = readFrame(vid);

% 3) Skip frames 
for k = 2 : 14
    if hasFrame(vid)
        readFrame(vid);
    else
        error('Video has fewer than 15 frames.');
    end
end

% 4) Read frame #15 (moving image)
frame15 = readFrame(vid);

% 5) Set parameters
levels    = 1;             % Number of pyramid levels
noi       = 15;            % Number of iterations
transform = 'affine';      % Warp model
delta_p_init = [1 0 0; ... % Identity (2x3) for affine
                0 1 0];

% 6) Call ECC/LK alignment
[results, results_lk, MSE, rho, MSELK] = ecc_lk_alignment( ...
    frame15, ...    
    frame1, ...     
    levels, ...     
    noi, ...        
    transform, ...  
    delta_p_init);  


disp('Final ECC warp (2x3):');
disp(results(1,15).warp);

disp('Final LK warp (2x3):');
disp(results_lk(1,15).warp);


disp('MSE (ECC) across iterations:');
disp(MSE);

disp('MSE (LK) across iterations:');
disp(MSELK);

disp('ECC correlation (rho) across iterations:');
disp(rho);
