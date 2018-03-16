file_path = 'E:/Structured_Light_Data/20180313/PlaneEpi/';

%% Load information
% fprintf('Loading...\n');
% frm_num = 20;
% x_pro_mats = cell(frm_num, 1);
% y_pro_mats = cell(frm_num, 1);
% image_mats = cell(frm_num, 1);
% pattern = double(imread([file_path, 'pattern_3size2color0.png']));
% for frm_idx = 1:frm_num
%   xfile_name = [file_path, 'cam_0/pro/xpro_mat', num2str(frm_idx-1), '.txt'];
%   yfile_name = [file_path, 'cam_0/pro/ypro_mat', num2str(frm_idx-1), '.txt'];
%   img_name = [file_path, 'cam_0/dyna/dyna_mat', num2str(frm_idx-1), '.png'];
%   x_pro_mats{frm_idx, 1} = load(xfile_name);
%   y_pro_mats{frm_idx, 1} = load(yfile_name);
%   image_mats{frm_idx, 1} = double(imread(img_name));
% end
% fprintf('\tLoad file finished.\n');

%% Interpolation
fprintf('Interpolation...\n');
for frm_idx = 1:frm_num
  for h = 2:1023
    for w = 2:1279
      x_pro = x_pro_mats{frm_idx, 1}(h, w);
      x_pro_lf = x_pro_mats{frm_idx, 1}(h, w-1);
      x_pro_rt = x_pro_mats{frm_idx, 1}(h, w+1);
      y_pro = y_pro_mats{frm_idx, 1}(h, w);
      y_pro_up = y_pro_mats{frm_idx, 1}(h-1, w);
      y_pro_dn = y_pro_mats{frm_idx, 1}(h+1, w);
      if x_pro < 0 && x_pro_lf > 0 && x_pro_rt > 0
        x_pro_mats{frm_idx, 1}(h, w) = (x_pro_lf + x_pro_rt) / 2;
      end
      if y_pro < 0 && y_pro_up > 0 && y_pro_dn > 0
        y_pro_mats{frm_idx, 1}(h, w)= (y_pro_up + y_pro_dn) / 2;
      end
    end
  end
end
fprintf('\tInterpolation finished.\n');

%% Fill points set
fprintf('Fill points set...\n');
points_set = cell(1024, 1280);
for h = 1:1024
  for w = 1:1280
    for frm_idx = 1:frm_num
      x_pro = x_pro_mats{frm_idx, 1}(h, w);
      y_pro = y_pro_mats{frm_idx, 1}(h, w);
      if x_pro > 0 && y_pro > 0
        points_set{h, w} = [points_set{h, w}; [x_pro, y_pro]];
      end
    end
  end
end
fprintf('\tFill points set finished.\n');

%% Calculate epi_AB_mat, error_mat
fprintf('Calculation...\n');
epi_A_mat = zeros(1024, 1280);
epi_B_mat = zeros(1024, 1280);
num_thred = 5;
for h = 1:1024
  for w = 1:1280
    points_num = size(points_set{h, w}, 1);
    if points_num >= num_thred
      AB_vec = points_set{h, w} \ ones(points_num, 1);
      epi_A_mat(h, w) = AB_vec(1);
      epi_B_mat(h, w) = AB_vec(2);
    end
  end
end
fprintf('\tCalculation finished.\n');

%% Calculate error_mat && mask_mat
fprintf('Calcualte error_mat...\n');
mask_mat = zeros(1024, 1280);
error_mat = ones(1024, 1280) * -1;
error_thred = 0.01;
for h = 1:1024
  for w = 1:1280
    epi_A = epi_A_mat(h, w);
    epi_B = epi_B_mat(h, w);
    if epi_A == 0 && epi_B == 0
      continue;
    end
    points_num = size(points_set{h, w}, 1);
    error_vec = ones(points_num, 1) - points_set{h, w}*[epi_A; epi_B];
    error_mat(h, w) = norm(error_vec);
    if error_mat(h, w) > error_thred
      mask_mat(h, w) = 0;
      epi_A_mat(h, w) = 0;
      epi_B_mat(h, w) = 0;
    else
      mask_mat(h, w) = 1;
    end
  end
end
fprintf('\tMask_mat & error_mat finished.\n');

%% Write result
fprintf('Saving...\n');
save('EpiMatA.txt', 'epi_A_mat', '-ascii');
save('EpiMatB.txt', 'epi_B_mat', '-ascii');
fprintf('\tSave result finished.\n');

%% Check
% x_test = 173;
% y_test = 249;
% frm_idx = 5;
% for w = 1:1280
%   epi_A = epi_A_mat(y_test, x_test);
%   epi_B = epi_B_mat(y_test, x_test);
%   x_pro = x_pro_mats{frm_idx, 1}(y_test, x_test);
%   y_pro = y_pro_mats{frm_idx, 1}(y_test, x_test);
%   h = -(epi_A/epi_B)*w + (1/epi_B);
%   pattern(uint32(h), w) = 255;
% end
% figure(1), imshow(pattern, [0, 255]);
% figure(2), imshow(image_mats{frm_idx, 1}, [0, 255]);