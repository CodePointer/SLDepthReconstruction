file_path = 'E:/Structured_Light_Data/20180313/Hand_2Color_2/';

ori_pattern = imread([file_path, 'pattern_3size2color0.png']);
d_pattern = double(ori_pattern) / 255;
set_pattern = d_pattern*100 + 20;
h = fspecial('gauss', 7, 1.2);
final_pattern = imfilter(set_pattern, h);
imshow(final_pattern, [0, 255]);
save([file_path, 'pattern_gauss.txt'], 'final_pattern', '-ascii');
% imwrite(uint8(final_pattern), 'pattern_gauss.png');