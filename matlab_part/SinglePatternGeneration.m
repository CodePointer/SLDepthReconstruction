pattern = zeros(800, 1280, 3);
dis_pattern = zeros(800, 1280, 3);
% gray_color = zeros(8, 1, 3);
% gray_color(:, 1, 1) = [0; 0; 0; 0; 1; 1; 1; 1];
% gray_color(:, 1, 2) = [0; 0; 1; 1; 1; 1; 0; 0];
% gray_color(:, 1, 3) = [0; 1; 1; 0; 0; 1; 1; 0];
% 
% for i = 1:8
%   pattern(:, (i-1)*4+1:i*4, 1) = gray_color(i, 1, 1);
%   pattern(:, (i-1)*4+1:i*4, 2) = gray_color(i, 1, 3);
%   pattern(:, (i-1)*4+1:i*4, 3) = gray_color(i, 1, 2);
% end
% 
% for w = 33:32:1280
%   pattern(:, w:w+31, :) = pattern(:, 1:32, :);
% end
% 
% imshow(pattern);
% imwrite(pattern, 'pattern_G8color4size0.png');
T = 72;
for w = 1:1280
  x = w / T * 2 * pi;
  pattern(:, w, 1) = (cos(x) + 1) / 2;
  if cos(x) > 0
    dis_pattern(:, w, 1) = 1;
  else
    dis_pattern(:, w, 1) = 0;
  end
  pattern(:, w, 2) = (cos(x + 2/3*pi) + 1) / 2;
  if cos(x + 2/3*pi) > 0
    dis_pattern(:, w, 2) = 1;
  else
    dis_pattern(:, w, 2) = 0;
  end
  pattern(:, w, 3) = (cos(x - 2/3*pi) + 1) / 2;
  if cos(x - 2/3*pi) > 0
    dis_pattern(:, w, 3) = 1;
  else
    dis_pattern(:, w, 3) = 0;
  end
end
for w = 3:12:1280
  dis_pattern(:, w:w+5, 1) = 0;
  dis_pattern(:, w:w+5, 2) = 0;
  dis_pattern(:, w:w+5, 3) = 0;
end
% imwrite(pattern, 'pattern_T36SC.png');
imwrite(dis_pattern, 'pattern_T36SD.png');