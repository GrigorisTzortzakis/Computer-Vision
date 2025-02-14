

% 1) Load the video
vid = VideoReader('video1_low.avi');

% 2) Read up to the first 100 frames into a cell array
maxFrames = 100;
frames = cell(1, maxFrames);
count = 0;
while hasFrame(vid) && count < maxFrames
    count = count + 1;
    frames{count} = readFrame(vid);
end

% If the video has fewer than 2 frames, we cannot do pairwise alignment
if count < 2
    error('Video has fewer than 2 frames!');
end

% We'll only align up to frame #count if count < 100
numFrames = count;  % actual number of frames read

% 3) Prepare arrays to store PSNR for each pair
PSNR_ECC = zeros(1, numFrames - 1);
PSNR_LK  = zeros(1, numFrames - 1);

% 4) Set alignment parameters (same for all pairs)
levels       = 1;                 % 1 pyramid level
noi          = 15;                % 15 iterations
transform    = 'affine';          % affine warp
delta_p_init = [1 0 0; 0 1 0];    % identity (2x3) for affine

% 5) Loop over each successive pair
for i = 1 : (numFrames - 1)
    
    % "Template" is frame i
    frameA = frames{i};
    % "Moving" image is frame i+1
    frameB = frames{i+1};
    
    % Call ECC + LK alignment
    [results, results_lk, MSE, ~, MSELK] = ecc_lk_alignment( ...
        frameB, ...      % moving
        frameA, ...      % template
        levels, ...      % 1
        noi, ...         % 15
        transform, ...   % 'affine'
        delta_p_init );  % identity init
    
    % Final MSE for ECC is MSE(noi); for LK it's MSELK(noi)
    finalMSE_ECC = MSE(noi);
    finalMSE_LK  = MSELK(noi);
    
    % Convert MSE -> PSNR (assuming max intensity = 255)
    PSNR_ECC(i) = 20 * log10(255 / finalMSE_ECC);
    PSNR_LK(i)  = 20 * log10(255 / finalMSE_LK);
    
    % Optional: Display progress
    fprintf('Frame %d -> %d: ECC PSNR=%.3f dB, LK PSNR=%.3f dB\n', ...
            i, i+1, PSNR_ECC(i), PSNR_LK(i));
end

% 6) Plot the PSNR results
figure;
plot(1:(numFrames-1), PSNR_ECC, '-o', 'LineWidth', 1.5, 'DisplayName', 'ECC');
hold on;
plot(1:(numFrames-1), PSNR_LK,  '-x', 'LineWidth', 1.5, 'DisplayName', 'LK');
hold off;
xlabel('Frame Pair (i -> i+1)');
ylabel('PSNR (dB)');
title('ECC vs. LK PSNR across successive frames');
legend('Location','best');
grid on;


