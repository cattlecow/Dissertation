clear all

%import data
fname = ['E:\论文\C1a.las']; %file path
lasReader = lasFileReader(fname);
points = readPointCloud(lasReader);
data = points.Location;

%plot points
figure
plot3(data(:,1),data(:,2),data(:,3),'.')
axis equal
title('All data')
print('-dpng', 'all_data.png'); 
close; 

%trim to central 1 m2
keep = data(:,1) > -0.1 & data(:,1) < 0.9 & data(:,2) > 0 & data(:,2) < 1;
data(~keep,:) = [];

figure
plot3(data(:,1),data(:,2),data(:,3),'.')
axis equal
title('Trimmed data') 
print('-dpng', 'trimmed_data.png'); 
close;  

%grid
res = 0.0005; %grid resolution
x_range = linspace(-0.1, 0.9, round(1 / res) + 1);  
y_range = linspace(0, 1, round(1 / res) + 1);  
[xgrid, ygrid] = meshgrid(x_range, y_range);
zgrid = griddata(data(:,1),data(:,2),data(:,3),xgrid,ygrid);

%plot
figure
surf(xgrid,ygrid,zgrid,'edgecolor','none','facecolor','w')
light
view([0 90])
axis equal
print('-dpng', 'white.png');  
title('white data')
close; 
figure
surf(xgrid,ygrid,zgrid,'edgecolor','none')
colormap(gray)
light
view([0 90])
axis equal
title('grey1 pic')
print('-dpng', 'grey.png');  
close; 
