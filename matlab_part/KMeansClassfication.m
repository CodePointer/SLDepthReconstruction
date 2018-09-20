% Used for image classification. Use k-means for classification.
% In HSV(H) channel.
dyna_path = [main_file_path, 'dyna/'];
class_path = [main_file_path, 'class_res/'];
mask_path = [main_file_path, 'mask_res/'];

img = imread([dyna_path, 'dyna_mat', num2str(frm_idx), '.png']);

center_intensity = [0, 42, 91, 127, 170, 230];

% Get mask
hsv_img = rgb2hsv(img);
raw_mask = imbinarize(hsv_img(:, :, 2), 0.1);
se = strel('disk',3);
raw_mask_dil = imdilate(raw_mask, se);
mask_final = imerode(raw_mask_dil, se);

% Hand-make delete:
mask_final(1:200, :) = 0;
mask_final(:, 1:600) = 0;
mask_final(870:end, :) = 0;

class_info = hsv_img(:, :, 1) * 255;
class_mask = zeros(1024, 1280);
while true
  new_center_intensity = zeros(1, 6);
  new_center_num = zeros(1, 6);
  for h = 1:1024
    for w = 1:1280
      if mask_final(h, w) == 0
        continue;
      end
      min_val = 256;
      min_idx = 0;
      for c = 1:6
        if min_val > abs(class_info(h, w) - center_intensity(c))
          min_val = abs(class_info(h, w) - center_intensity(c));
          min_idx = c;
        end
      end
      class_mask(h, w) = min_idx;
      new_center_intensity(min_idx) = new_center_intensity(min_idx) + class_info(h, w);
      new_center_num(min_idx) = new_center_num(min_idx) + 1;
    end
  end
  new_center_intensity = new_center_intensity ./ new_center_num;
  if norm(new_center_intensity - center_intensity) < 0.1
    break;
  else
    center_intensity = new_center_intensity;
  end
end
imshow(class_mask, [0, 6]);
imwrite(uint8(class_mask*10), [class_path, 'class', num2str(frm_idx), '.png']);
imwrite(mask_final, [mask_path, 'mask', num2str(frm_idx), '.png']);