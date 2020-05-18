% By Isaac Chen in 2019, with the inspiration from Prof. Hongdong Li, ANU

function main()
k=10; %number of clusters
im = imread('peppers.png');
%im = imread('mandm.png');
imlab = rgb2lab(im); %convert rgb to lab
[r,c,~] = size(im);
data5 = zeros(r*c,5); %initialise the datapoints for 5-d vector (i.e the ones with coordinates)
data3 = zeros(r*c,3); %initialise the datapoints for 3-d vector (i.e the ones without coordinates)
xandy = zeros(r*c,2); %will be used to store the coordinates of each datapoint
counter = 1;
for y = 1:r
    for x = 1:c
        lab = imlab(y,x,:);
        xandy(counter,:) = [x y];
        %normalise the datapoints so that each dimension affect the clustering equally
        L=lab(1);
        A=100*(lab(2)+100)/200;
        B=100*(lab(3)+100)/200;
        X=100*x/c;
        Y=100*y/r;
        data5(counter,:)=[L A B X Y]; %convert the image to a matrix of datapoint with with coordinates. 
        %each row stands for an image and all columns respectively stand for L, a*, b*, x and y coordinates 
        data3(counter,:)=[L A B]; %convert the image to a matrix of datapoint with without coordinates.
        %each row stands for an image and all columns respectively stand for L, a*, b*
        counter = counter + 1;
    end
end
% both set of datapoints are processed by normal k-mean clustering and k-mean++

[clusters5,centroids5]= my_kmeans(data5,k);
[clusters5_pp,centroids5_pp] = my_kmeans_pp(data5,k);
[clusters3,centroids3] = my_kmeans(data3,k);
[clusters3_pp,centroids3_pp] = my_kmeans_pp(data3,k);

im_s=zeros(r,c,3); %initialise the image of segmentation
for i=1:k
    clt = clusters5{i}; %get all points from a cluster
    [s,~]=size(clt);
    labcolor=centroids5(i,:); %get the color of the corresponding centroid
    %denormalise
    newl=labcolor(1);
    newa=(labcolor(2)*200/100)-100;
    newb=(labcolor(3)*200/100)-100;
    rgbv=lab2rgb([newl newa newb]); %convert lab value back to rgb
    %all pixels in the same cluster are given the color of its centroid
    for j=1:s
        inx = clt(j);%get the index of the datapoint
        ycod=xandy(inx,2);%find out its x and y coordinates
        xcod=xandy(inx,1);
        im_s(ycod,xcod,:)=rgbv;%change the cooresponding pixel in the image of segmentation 
    end
end
figure()
imagesc(im_s);
title("normal k-mean; with coordinates; k="+string(k));

%similar procedure is done for k-mean++ and the input has coordinates
im_s=zeros(r,c,3);
for i=1:k
    clt = clusters5_pp{i};
    [s,~]=size(clt);
    labcolor=centroids5_pp(i,:);
    newl=labcolor(1);
    newa=(labcolor(2)*200/100)-100;
    newb=(labcolor(3)*200/100)-100;
    rgbv=lab2rgb([newl newa newb]);
    for j=1:s
        inx = clt(j);
        ycod=xandy(inx,2);
        xcod=xandy(inx,1);
        im_s(ycod,xcod,:)=rgbv;
    end
end
figure()
imagesc(im_s);
title("k-mean++; with coordinates; k="+string(k));

%similar procedure is done for normal k-mean and the input has no coordinate
im_s=zeros(r,c,3);
for i=1:k
    clt = clusters3{i};
    [s,~]=size(clt);
    labcolor=centroids3(i,:);
    newl=labcolor(1);
    newa=(labcolor(2)*200/100)-100;
    newb=(labcolor(3)*200/100)-100;
    rgbv=lab2rgb([newl newa newb]);
    for j=1:s
        inx = clt(j);
        ycod=xandy(inx,2);
        xcod=xandy(inx,1);
        im_s(ycod,xcod,:)=rgbv;
    end
end
figure()
imagesc(im_s);
title("normal k-mean; without coordinates; k="+string(k));

%similar procedure is done for k-mean++ and the input has no coordinate
im_s=zeros(r,c,3);
for i=1:k
    clt = clusters3_pp{i};
    [s,~]=size(clt);
    labcolor=centroids3_pp(i,:);
    newl=labcolor(1);
    newa=(labcolor(2)*200/100)-100;
    newb=(labcolor(3)*200/100)-100;
    rgbv=lab2rgb([newl newa newb]);
    for j=1:s
        inx = clt(j);
        ycod=xandy(inx,2);
        xcod=xandy(inx,1);
        im_s(ycod,xcod,:)=rgbv;
    end
end
figure()
imagesc(im_s);
title("k-mean++; without coordinates; k="+string(k));
end


%normal k-mean function
%input:datapoint matrix and k
%output:clusters of datapoints and all centroids
function [result,centroids] = my_kmeans(datapoints, k)
tic;
[r,c] = size(datapoints);
%step 1: randomly select k datapoints as centroids
centroids = zeros(k,c);
cent_num = randi(r,k,1);
for i=1:k
    centroids(i,:)=datapoints(cent_num(i),:);
end

clusters = zeros(r,1); % initialise clusters indicator. i.e each datapoint has a cluster number(centroid index)
%step 2: find out the closest centroid for all datapoints
while 1
    for i = 1:r
        min_dist = 9999999; %initilise the minimum distance between the datapoint and its centroid
        current_centroid = 0; 
        for j=1:k
            cent = centroids(j,:); %get a centroid
            datapt = datapoints(i,:); %get a datapoint
            %calculate the distance between the datapoint and the centroid
            dist = 0;
            for d=1:c
                dist = dist + power(cent(d)-datapt(d),2);
            end
            dist = sqrt(dist);
            %if the distance is less than the current minimum distance
            %then the current minimum distance is replaced with the distance and its closest centroid is changed coorespoindly
            if dist < min_dist
                min_dist = dist;
                current_centroid = j;
            end
        end
        clusters(i)=current_centroid; %so at the end we can find out the closest centroid for all datapoints
        
    end
    previous_centroids = centroids; %save the current centroids
    %step 3: move all centroids to the center of its cluster
    for i = 1:k
        for j = 1:c
            total_value = 0;%initialise the sum of value in the same dimension for each cluster
            counter = 0; %counter the number of datapoints of the cluster
            for l= 1:r
                %get the number of datapoints for all clusters
                %and calculate the sum of value for all dimensions and for all clusters
                if clusters(l)==i
                    datapt = datapoints(l,:);
                    total_value = total_value + datapt(j);
                    counter = counter + 1;
                end
            end
            % if there is a centroid has no datapoint at all, it stays still
            % otherwise it is moved to the center of its datapoints (the average value)
            if counter ==0
                centroids(i,j) = centroids(i,j);
            else
                centroids(i,j)=floor(total_value/counter);
            end
        end
    end
    %step 4: repeat step 2 and 3 until the positions of all centroids don't change anymore
    if (previous_centroids==centroids)
        break;
    end
end
result = {};
%based on the cluster indicator, group all datapoints (indices) in a cell array
for i = 1:k
    counter1 = 0; %counter the number of datapoint for one cluster
    for j = 1:r
        if clusters(j) ==i
            counter1 = counter1+1;
        end
    end
    clt = zeros(counter1,1); % initialise the cluster
    counter2 = 0;
    for j = 1:r
        if clusters(j) ==i
            counter2 = counter2+1;
            clt(counter2)=j; % put all datapoint indices to the cluster
        end
    end
    result{end+1} = clt; %result is the clusters with grouped datapoint indices
end
disp(centroids);
disp("k_means:   "+toc);
end

%k-mean++ function
%input:datapoint matrix and k
%output:clusters of datapoints and all centroids
function [result,centroids] = my_kmeans_pp(datapoints, k)
tic;
[r,c] = size(datapoints);
%step 1: find k centroids that are far from each other
centroids = zeros(k,c);
D = zeros(r,1);
probability_D = zeros(1,r); %initialise the probability for later selecting centroids
first_centroid = datapoints(randi(r),:);%randomly select a datapoint as the first centroid
cent_counter = 1;
centroids(cent_counter,:)=first_centroid;

while cent_counter < k
    %based on the current chosen centroids, find out the distances between
    %all datapoints and its closest centroid
    for i=1:r
        datapt = datapoints(i,:);
        min_dist = 9999999;
        for j=1:cent_counter
            dist = 0;
            cen = centroids(j,:);
            for d=1:c
                dist = dist + power(cen(d)-datapt(d),2);
            end
            dist = sqrt(dist);
            if dist<min_dist
                min_dist=dist;
            end
        end
        D(i)=min_dist; %store the distance between the datapoint and its closest centroid
    end
    
    D_total_value = 0;
    for i = 1:r
        D_total_value = D_total_value + power(D(i),2);%sum all shortest distances
    end
    for i = 1:r
        p = power(D(i),2);
        prob = p/D_total_value;
        probability_D(i)=prob; 
        %the probability of the datapoint being selected is based on its distance to its closest centroid
        %which mean if the datapoints that are far from all centroids are likely to be selected as another centroid
    end
    distribution=cumsum([0, probability_D]);
    new_cent = sum(rand >= distribution); %select centroid based on the above probabilities
    new_centroid = datapoints(new_cent,:);
    cent_counter = cent_counter+1;
    centroids(cent_counter,:)=new_centroid; %so that at the end we have k centroids that are far from each other 
end

%the remain steps are similar with the normal k-mean function above
%step 2: find out the closest centroid for all datapoints
clusters = zeros(r,1);
while 1
    for i = 1:r
        min_dist = 9999999;
        current_centroid = 0;
        for j=1:k
            cent = centroids(j,:);
            datapt = datapoints(i,:);
            dist = 0;
            for d=1:c
                dist = dist + power(cent(d)-datapt(d),2);
            end
            dist = sqrt(dist);
            if dist < min_dist
                min_dist = dist;
                current_centroid = j;
            end
        end
        clusters(i)=current_centroid;        
    end
    previous_centroids = centroids;
    %step 3: move all centroids to the center of its cluster
    for i = 1:k
        for j = 1:c
            total_value = 0;
            counter = 0;
            for l= 1:r
                if clusters(l)==i
                    datapt = datapoints(l,:);
                    total_value = total_value + datapt(j);
                    counter = counter + 1;
                end
            end
            if counter ==0
                centroids(i,j) = centroids(i,j);
            else
                centroids(i,j)=floor(total_value/counter);
            end
        end
    end
    %step 4: repeat step 2 and 3 until the positions of all centroids don't change anymore
    if (previous_centroids==centroids)
        break;
    end
end
result = {};
%based on the cluster indicator, group all datapoints (indices) in a cell array
for i = 1:k
    counter1 = 0;
    for j = 1:r
        if clusters(j) ==i
            counter1 = counter1+1;
        end
    end
    clt = zeros(counter1,1);
    counter2 = 0;
    for j = 1:r
        if clusters(j) ==i
            counter2 = counter2+1;
            clt(counter2)=j;
        end
    end
    result{end+1} = clt;
end
disp(centroids);
disp("k_means++: "+ toc);
end






