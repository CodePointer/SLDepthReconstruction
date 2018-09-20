pro_path = 'E:/SLDataSet/20180626/HandMove_BlkDis/pro/';
dyna_path = [main_file_path, 'dyna/'];
class_path = [main_file_path, 'class_res/'];
mask_path = [main_file_path, 'mask_res/'];
x_range_path = [main_file_path, 'pro_range_res/'];

pro_mat = load([pro_path, 'xpro_mat', '0', '.txt']);
mask_mat = imread([mask_path, 'mask', '0', '.png']);
class_mat = imread([class_path, 'class', '0', '.png']);
dyna_mat = imread([dyna_path, 'dyna_mat', '0', '.png']);

% Fill initial pro_mat
x_pro_mat = pro_mat;
flag = true;
while flag
  flag = false;
  tmp_mat = x_pro_mat;
  for h = 1:1024
    for w = 1:1280
      if mask_mat(h, w) == 0
        continue;
      end
      if x_pro_mat(h, w) > 0
        for nh = h-1:h+1
          for nw = w-1:w+1
            if nw > 1280
              continue;
            end
            if tmp_mat(nh, nw) < 0
              tmp_mat(nh, nw) = x_pro_mat(h, w);
              flag = true;
            end
          end
        end
      end
    end
  end
  x_pro_mat = tmp_mat;
end

% Calculate depth range
last_x_pro_range = x_pro_mat;
last_mask_mat = mask_mat;
for frm_idx = 1:100
  x_pro_range = zeros(1024, 1280);
  dyna_mat = imread([dyna_path, 'dyna_mat', num2str(frm_idx), '.png']);
  mask_mat = imread([mask_path, 'mask', num2str(frm_idx), '.png']);
  class_mat = imread([class_path, 'class', num2str(frm_idx), '.png']);
  for h = 1:1024
    for w = 1:1280
      if mask_mat(h, w) == 0
        continue;
      end
      % Get last x_pro
      tmp_vec = zeros(25, 1);
      valid_num = 0;
      for nh = h-5:h+5
        for nw = w-5:w+5
          if nw > 1280
            continue;
          end
          if last_mask_mat(nh, nw) == 0
            continue;
          end
          valid_num = valid_num + 1;
          tmp_vec(valid_num) = last_x_pro_range(nh, nw);
          if tmp_vec(valid_num) == 0
            continue;
          end
        end
      end
      last_x_pro = median(tmp_vec(1:valid_num));
      if isnan(last_x_pro)
        for nh = h:-1:1
          if mask_mat(h, w) == 1
            last_x_pro = x_pro_range(nh, w);
            break;
          end
        end
        continue;
      end
      
      % Get period
      class_num = double(7 - class_mat(h, w) / 10);
      left_value = floor((last_x_pro - 12*class_num) / 72) * 72 + 12*class_num;
      right_value = ceil((last_x_pro - 12*class_num) / 72) * 72 + 12*class_num;
      if abs(last_x_pro - left_value) < abs(right_value - last_x_pro)
        x_pro_range(h, w) = left_value;
      else
        x_pro_range(h, w) = right_value;
      end
      if x_pro_range(h, w) < 100 || isnan(x_pro_range(h, w))
        continue;
      end
    end
  end
  fprintf('frm_idx=%d\n', frm_idx); 
  last_x_pro_range = x_pro_range;
  last_mask_mat = mask_mat;
  imshow(x_pro_range, [300, 1280]);
  imwrite(uint16(x_pro_range), [x_range_path, 'xpro_range', num2str(frm_idx), '.png']);
end
