warning('off');

node_set = load(['./result/node', num2str(1), '.txt']);
node_mat = double(dyna_mat) / 255;
  for i = 1:5120
    if node_set(i, 1) == 1
      depth = node_set(i, 2);
      pos_x = node_set(i, 3); pos_y = node_set(i, 4);
      node_mat(pos_y:pos_y+2, pos_x:pos_x+2, :) = 1.0;
    end
  end
  
for frm_idx = 1:90
  % Load mats
  depth_mat = load(['./result/depth', num2str(frm_idx), '.txt']);
  dyna_mat = imread(['./dyna/dyna_mat', num2str(frm_idx), '.png']);
  x_pro_range = imread(['./result/x_pro_range', num2str(frm_idx), '.png']);
  x_pro_mat = load(['./result/x_pro', num2str(frm_idx), '.txt']);
  img_class = imread(['./class_res/class', num2str(frm_idx), '.png']) / 10;
  
  
  % dyna_mat
  part_dyna_mat = double(dyna_mat(:, 601:end, :)) / 255;
  
  % img_class
  part_img_class = zeros(size(part_dyna_mat));
  for h = 1:1024
    for w = 1:680
      class_num = img_class(h, w + 600);
      if class_num == 6 || class_num == 1 || class_num == 2
        part_img_class(h, w, 1) = 1.0;
      end
      if class_num == 2 || class_num == 3 || class_num == 4
        part_img_class(h, w, 2) = 1.0;
      end
      if class_num == 4 || class_num == 5 || class_num == 6
        part_img_class(h, w, 3) = 1.0;
      end
    end
  end
  
  % x_pro_mat, x_pro_range_mat
  part_x_pro_mat = zeros(size(part_dyna_mat));
  part_range_mat = zeros(size(part_dyna_mat));
  min_val = 200; max_val = 1100;
  for h = 1:1024
    for w = 1:680
      if x_pro_mat(h, w+600) > 0
        pro_value = x_pro_mat(h, w+600);
        range_value = double(x_pro_range(h, w+600));
        part_x_pro_mat(h, w, 1) = 1.0;
        part_x_pro_mat(h, w, :) = (pro_value - min_val) / (max_val - min_val);
        part_range_mat(h, w, 1) = 1.0;
        part_range_mat(h, w, :) = (range_value - min_val) / (max_val - min_val);
      end
    end
  end
  
  % depth
  min_depth = 12; max_depth = 30;
  color_bar = jet(500);
  part_depth_mat = zeros(size(part_dyna_mat));
  for h = 1:1024
    for w = 1:680
      if depth_mat(h, w+600) > 0
        depth_value = depth_mat(h, w+600);
        color_idx = uint32(500 * (depth_value - min_depth) / (max_depth - min_depth));
        if color_idx <= 0
          color_idx = 1;
        end
        if color_idx > 500
          color_idx = 500;
        end
        part_depth_mat(h, w, 1) = color_bar(color_idx, 1);
        part_depth_mat(h, w, 2) = color_bar(color_idx, 2);
        part_depth_mat(h, w, 3) = color_bar(color_idx, 3);
      end
    end
  end
  
  total_mat = [part_dyna_mat, part_img_class, part_x_pro_mat, part_depth_mat];
  imshow(total_mat);
  
  [A, map] = rgb2ind(total_mat, 256);
  if frm_idx == 1
    imwrite(A, map, 'show.gif', 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
  else
    imwrite(A, map, 'show.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
  end
  
  fprintf('%d finished.\n', frm_idx);
end