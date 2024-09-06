clc
clear all
close all
% pointCloudMy=importdata('Glenderamackin_PointCloud.las');
pointCloudMy=pcread('AfonLledr_PointCloud.ply'); %point cloud and this code in the same folder
% pointCloudMy=pcdownsample(pointCloudMy,"random",0.9);
% figure;pcshow(pointCloudMy.Location);

Eular=[0,0,180];% Z Y  X
Eular=Eular.*pi/180;
p_x=0;
p_y=0;
p_z=50; 
R_PointCloudCrood2Cam=eul2rotm(Eular);
t_PointCloudCrood2Cam=[p_x;p_y;p_z];
T_PointCloudCrood2Cam=[R_PointCloudCrood2Cam,t_PointCloudCrood2Cam;[0,0,0,1]];

AxisLength=10;
% figure;
hold on;plot3([0;AxisLength],[0;0],[0;0],'Color','r','LineWidth',2);
hold on;plot3([0;0],[0;AxisLength],[0;0],'Color','g','LineWidth',2);
hold on;plot3([0;0],[0;0],[0;AxisLength],'Color','b','LineWidth',2);

CroodPoint=[0,0,0;1,0,0;0,1,0;0,0,1];
CroodPoint=CroodPoint.*AxisLength;
CroodPointQi=[CroodPoint,ones(size(CroodPoint,1),1)];
CroodPointInCam=T_PointCloudCrood2Cam*CroodPointQi';
CroodPointInCam=CroodPointInCam(1:3,:)';
hold on;plot3([CroodPointInCam(1,1);CroodPointInCam(2,1)],[CroodPointInCam(1,2);CroodPointInCam(2,2)],[CroodPointInCam(1,3);CroodPointInCam(2,3)],'Color','r','LineWidth',2);
hold on;plot3([CroodPointInCam(1,1);CroodPointInCam(3,1)],[CroodPointInCam(1,2);CroodPointInCam(3,2)],[CroodPointInCam(1,3);CroodPointInCam(3,3)],'Color','g','LineWidth',2);
hold on;plot3([CroodPointInCam(1,1);CroodPointInCam(4,1)],[CroodPointInCam(1,2);CroodPointInCam(4,2)],[CroodPointInCam(1,3);CroodPointInCam(4,3)],'Color','b','LineWidth',2);



PointInCam=T_PointCloudCrood2Cam*[pointCloudMy.Location,ones(size(pointCloudMy.Location,1),1)]';
PointInCam=PointInCam';
PointInCam=PointInCam(:,1:3);

% virtual camera, adjust the parameters here
f=50;
BaMianH=3;
BaMianW=4;
dx=0.001;
dy=0.001;
ImgHeight=BaMianH/dy;
ImgWidth=BaMianW/dx;
K=[f/dx,0,ImgWidth/2;0,f/dy,ImgHeight/2;0,0,1];
ImgColor=zeros(ImgHeight,ImgWidth,3);
ImgColor=uint8(ImgColor);
ImgDepth=zeros(ImgHeight,ImgWidth,1);
for i=1:size(PointInCam,1)
    uvz=K*PointInCam(i,:)';
    uv1=uvz./uvz(3);
    uv1=round(uv1);
    if uv1(2)<1||uv1(1)<1||uv1(1)>ImgWidth||uv1(2)>ImgHeight
        continue;
    end
    ImgDepth(uv1(2),uv1(1))=uvz(3);
    ImgColor(uv1(2),uv1(1),1)=pointCloudMy.Color(i,1);
    ImgColor(uv1(2),uv1(1),2)=pointCloudMy.Color(i,2);
    ImgColor(uv1(2),uv1(1),3)=pointCloudMy.Color(i,3);
end
Data=ImgDepth(ImgDepth~=0);
ImgDepth=(ImgDepth-min(Data))./(max(Data)-min(Data));
ImgDepth1=medfilt2(ImgDepth,[11,11]);
figure;imshow(ImgDepth1,[0,1]);
imwrite(ImgDepth1, 'ImgDepth1.jpg');
ImgColor=imresize(ImgColor,0.25);
ImgColor=imresize(ImgColor,4);
figure;imshow(ImgColor);
imwrite(ImgColor, 'ImgColor.jpg');




