clc;
N = 20;
TrainingMatrix = [];

for j=1:N
I = ['face',num2str(j),'.jpg'];
i =imread(I);
i = imresize(i,[400 500]);
detect = vision.CascadeObjectDetector;
bbox = step(detect,i);
nRows = size(bbox, 1);
% for xp=1:size(i,1)
%     for yp=1:size(i,2)
%        
%             i(xp,yp,:) = [255,255,255];
%     end
% end
for n=1:nRows
for xp=1:size(i,1)
    for yp=1:size(i,2)
        if xp>bbox(n,2) && yp>bbox(n,1) && xp<bbox(n,2)+bbox(n,4) && yp <bbox(n,1)+bbox(n,3)
            gs = 0;
            i(xp,yp,:) = [gs,gs,gs];
        end
    end
end
end
i = imresize(i,[200, 200]);
i = im2bw(i);
tra = reshape(i,1,40000);
tra =[tra nRows];
TrainingMatrix = [TrainingMatrix; tra;];
disp(nRows);
%figure; 
imshow(i);
end
Group = [1;1;1;1;1;1;1;1;1;1
         2;2;2;2;2;2;2;2;2;2];
     
     

i =imread('samp2.jpg');
i = imresize(i,[400 500]);
detect = vision.CascadeObjectDetector;
bbox = step(detect,i);
nRows = size(bbox, 1);
for xp=1:size(i,1)
    for yp=1:size(i,2)
       
            i(xp,yp,:) = [255,255,255];
    end
end
for n=1:nRows
for xp=1:size(i,1)
    for yp=1:size(i,2)
        if xp>bbox(n,2) && yp>bbox(n,1) && xp<bbox(n,2)+bbox(n,4) && yp <bbox(n,1)+bbox(n,3)
            gs = 0;
            i(xp,yp,:) = [gs,gs,gs];
        end
    end
end
end
i = imresize(i,[200, 200]);
i = im2bw(i);
samp = reshape(i,1,40000);
samp =[samp nRows];


Class = knnclassify(samp, TrainingMatrix, Group);

disp(Class);
    

