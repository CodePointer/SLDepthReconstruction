file_path = 'E:/Structured_Light_Data/20180313/PlaneEpi/';

% Load informations
frm_num = 1;
x_pro_mats = cell(frm_num, 1);
y_pro_mats = cell(frm_num, 1);
image_mats = cell(frm_num, 1);
pattern = double(imread([file_path, 'pattern_3size2color0.png']));
for frm_idx = 1:frm_num
  xfile_name = [file_path, 'cam_0/pro/xpro_mat', num2str(frm_idx-1), '.txt'];
  yfile_name = [file_path, 'cam_0/pro/ypro_mat', num2str(frm_idx-1), '.txt'];
  img_name = [file_path, 'cam_0/dyna/dyna_mat', num2str(frm_idx-1), '.png'];
  x_pro_mats{frm_idx, 1} = load(xfile_name);
  y_pro_mats{frm_idx, 1} = load(yfile_name);
  image_mats{frm_idx, 1} = double(imread(img_name));
end
fprintf('Load file finished.\n');

%% Calculate FundatmentalMatrix F
% Fill mats (only use 1 frame)
matched_points_cam = [];
matched_points_pro = [];
for frm_idx = 1:1
  points_cam = zeros(1024*1280, 2);
  points_pro = zeros(1024*1280, 2);
  valid_num = 0;
  for h = 1:1024
    for w = 1:1280
      x_pro = x_pro_mats{frm_idx, 1}(h, w);
      y_pro = y_pro_mats{frm_idx, 1}(h, w);
      if x_pro < 0 || y_pro < 0
        continue;
      end
      valid_num = valid_num + 1;
      x_c = w - 1;
      y_c = h - 1;
      x_p = x_pro;
      y_p = y_pro;
      points_cam(valid_num, :) = [x_c, y_c];
      points_pro(valid_num, :) = [x_p, y_p];
    end
  end
  matched_points_cam = [matched_points_cam, points_cam(1:valid_num, :)];
  matched_points_pro = [matched_points_pro, points_pro(1:valid_num, :)];
end
% fprintf('Fill A_mat finished.\n');
% 
% % Calculate F mat
% [~, ~, V] = svd(A_mat, 0);
% F_vec = V(:, end);
% F_raw = reshape(F_vec, 3, 3)';
% [U, S, V] = svd(F_raw);
% S(3, 3) = 0;
% F_mat = U*S*V;
% F_mat = F_mat / F_mat(3, 3);
F_mat = estimateFundamentalMatrix(matched_points_cam, matched_points_pro);

% Test
x_test = 500;
y_test = 600;
epi_vec = [x_test, y_test, 1]*F_mat;
for w = 1:1280
  h = -(epi_vec(1)/epi_vec(2))*w-(epi_vec(3)/epi_vec(2));
  if h >= 1 && h <= 800
    pattern(uint32(h), w) = 128;
  end
end