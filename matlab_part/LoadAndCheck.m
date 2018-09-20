main_path = './result_0809/';

for frm_idx = 15:15
  fprintf('frm_idx = %d\n', frm_idx);
  % Load depth
  depth_mat = load([main_path, 'depth', num2str(frm_idx), '.txt']);
  node_set = load([main_path, 'node', num2str(frm_idx), '.txt']);
  x_pro_range = imread([main_path, 'x_pro_range', num2str(frm_idx), '.png']);
  img_class = imread(['./class_res/class', num2str(frm_idx), '.png']) / 10;
  mask_mat = imread(['./mask_res/mask', num2str(frm_idx), '.png']);
%   last_depth_mat = load([main_path, 'depth', num2str(frm_idx - 1), '.txt']);
%   last_node_set = load([main_path, 'node', num2str(frm_idx -  1), '.txt']);
%   last_x_pro_range = imread([main_path, 'x_pro_range', num2str(frm_idx - 1), '.png']);
%   last_img_class = imread(['./class_res/class', num2str(frm_idx - 1), '.png']) / 10;
%   last_mask_mat = imread(['./mask_res/mask', num2str(frm_idx - 1), '.png']);
  
  % Show node_set:
  node_mat = zeros(1024, 1280);
  for i = 1:5120
    if node_set(i, 1) == 1
      depth = node_set(i, 2);
      pos_x = node_set(i, 3); pos_y = node_set(i, 4);
      node_mat(pos_y + 1, pos_x + 1) = depth;
    end
  end
%   last_node_mat = zeros(1024, 1280);
%   for i = 1:5120
%     if last_node_set(i, 1) == 1
%       depth = last_node_set(i, 2);
%       pos_x = last_node_set(i, 3); pos_y = last_node_set(i, 4);
%       last_node_mat(pos_y + 1, pos_x + 1) = depth;
%     end
%   end
  for h = 1:1024
    for w = 1:1280
      if mask_mat(h, w) == 1 && x_pro_range(h, w) == 0
        fprintf('%d: (%d, %d)\n', frm_idx, h, w); 
        % TODO: Add fill matrix method, flood fill, for x_pro_range mat.
        % The empty value in mask mat field will influence the period
        % prediction problem.
        % 11: (411, 755)
%         11: (411, 756)
%         11: (412, 755)
%         11: (412, 756)
%         11: (413, 755)
%         11: (413, 756)
%         11: (414, 755)
%         11: (414, 756)
%         11: (415, 755)
%         11: (415, 756)
%         11: (416, 755)
%         11: (416, 756)
%         11: (417, 755)
%         11: (417, 756)
%         11: (418, 755)
%         11: (418, 756)
%         11: (419, 755)
%         11: (419, 756)
%         11: (710, 1079)
%         11: (759, 763)
%         11: (759, 764)
%         11: (792, 1091)
%         11: (793, 1091)
%         11: (794, 1091)
%         11: (795, 1091)
      end
    end
  end
%   figure(1), imshow(depth_mat, [15, 25]);
%   figure(2), imshow(mask_mat)
%   figure(3), imshow(x_pro_range, [100, 1200])
end