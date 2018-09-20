img = imread([file_path, 'dyna_mat', num2str(frm_idx), '.png']);

channal_thred = [50, 100, 100];

bw_img = zeros(size(img));
fixed_img = zeros(size(img));
for c = 1:3
  bw_img(:, :, c) = imbinarize(double(img(:, :, c)), channal_thred(c));
  se = strel('disk',c);
  dil_mat = imdilate(bw_img(:, :, c), se);
  fixed_img(:, :, c) = imerode(dil_mat, se);
end
imshow(fixed_img);