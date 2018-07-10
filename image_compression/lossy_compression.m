clear
a=double(imread('mandrill-small.tiff'));
a=permute(a,[1 3 2]);
b=reshape(a,3,128*128);
x=randi([1 16384],1,16);
centroid=b(:,x);
oldcentroid=centroid-centroid;
while norm(centroid-oldcentroid)>1e-9
score=zeros(4,16);
beta=dot(centroid,centroid);
oldcentroid=centroid;
for i=1:1:128*128
 [value,index]=min(-2.*centroid'*b(:,i)+beta');
 score(1:3,index)=((1/(score(4,index)+1)).*(b(:,i)))+((score(4,index)/(score(4,index)+1)).*(score(1:3,index)));
 score(4,index)=score(4,index)+1;
end
centroid=score(1:3,:);
end
centroid=round(centroid);
anew=double(imread('mandrill-large.tiff'));
anew=permute(anew,[1 3 2]);
bnew=reshape(anew,3,512*512);
for i=1:1:512*512
 [value,index]=min(-2.*centroid'*bnew(:,i)+beta');
bnew(:,i)=centroid(:,index);
end
newb=(reshape(bnew,512,3,512));
newb=permute(newb,[1,3,2]);
imshow(uint8(newb));

