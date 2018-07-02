function adaboost()

%flags
flag_data_subset = 1;
flag_extract_features = 1;
flag_parpool = 1;
flag_boosting = 1;
flag_top_20_haar_filters = 1;
flag_plot_train_error = 1;
flag_plot_train_error_weak_classifier = 1;
flag_plot_histograms = 1;
flag_roc_curve = 1;

%parallel processing using parpool
if flag_parpool
     delete(gcp('nocreate'));
     myCluster = parcluster('local');
     myCluster.NumWorkers = 4;  	%number of logical cores
     saveProfile(myCluster);    	%'local' profile now updated
     parpool(4,'IdleTimeout', Inf); %only 4 logical cores in my laptop; no timeout
end

%unit tests
test_sum_rect();
test_filters();

%number of images
if flag_data_subset
    N_pos = 500;
    N_neg = 500;
else
    N_pos = 11838;
    N_neg = 45356;
    
    %N_pos = 11838;
    %N_neg = 25356;
end
N = N_pos + N_neg;
w = 16;
h = 16;

%----------------------------------------------------------------------------------
%%task 1- haar filters
%----------------------------------------------------------------------------------

%loading images
if flag_extract_features
    tic;
    I = zeros(N, h, w);
    for i=1:N_pos
        I(i,:,:) = rgb2gray(imread(sprintf('newface16/face16_%06d.bmp',i), 'bmp'));
    end
    for i=1:N_neg
        I(N_pos+i,:,:) = rgb2gray(imread(sprintf('nonface16/nonface16_%06d.bmp',i), 'bmp'));
    end
    fprintf('Loading images took %.2f secs.\n', toc);
end

%constructing filters
A = filters_A();
B = filters_B();
C = filters_C();
D = filters_D();

%selecting number of filters
if flag_data_subset
    filters = [A(1:250,:); B(1:250,:); C(1:250,:); D(1:250,:)];
    %filters = [A(1:3276,:); B(1:3276,:); C(1:1716,:); D(1:900,:)];
else
    filters = [A; B; C; D];
end

%saving filter matrix
save('filters.mat', 'filters');


T = 101;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%extract features
if flag_extract_features
    tic;
    I = normalize(I);
    II = integral(I);
    features = compute_features(II, filters);
    clear I;
    clear II;
    save('features.mat', '-v7.3', 'features');
    fprintf('Extracting %d features from %d images took %.2f secs.\n', size(filters, 1), N, toc);
else
    load('features.mat','features');
end

%performing boosting
if(flag_boosting == 1)
    fprintf('Running AdaBoost with %d features from %d images.\n', size(filters, 1), N);
    tic;
    %% implement this
    w = ones(1, N)*(1.0/N);	%image weight vectors
    T = 101; 				%total number of weak classifiers selected

    y = ones(1, N);					 %initialising y matrix
    y(1, N_pos+1:N_pos+N_neg) = -1;  %y labels for data
	
	wclassifier_index = zeros(1, T);			%initialising y matrix
    wclassifier = zeros(size(features,1),3);	%stores theta, polarity, alphas
    
	%iterating over selected weak classifiers	
    for t = 1:T
        
		%for each weak classifier, compute theta i.e. minimum error
        parfor w_i = 1:size(features,1)
	    %for w_i = 1:size(features,1)	
            
			if ismember(w_i, wclassifier_index)
                continue;
            end
            min_theta = min(features(w_i,:));						%obtaining min error for given weak classifier
            max_theta = max(features(w_i,:));						%obtaining max error for given weak classifier
            possible_thetas = linspace(min_theta, max_theta, 50);	%interpolating theta/error b/w min & max in 50 bins
            
			min_error = abs(log(0));	%setting to infinity
            s = 1; 						%polarity value
			
            %for computed min theta value for each weak classifier
			for theta_index = 1:size(possible_thetas,2)
                no_of_ones = zeros(1, N);
                
				%iterating over all images
                for image_no=1:size(features,2)
                    
					%face -> 1, non-face -> -1
					if features(w_i, image_no) < possible_thetas(theta_index) 
                        h = -1*s;
                        %disp('It is a face');
                    else
                        h = +1*s;
                        %disp('It is a non-face');
                    end
                    
					%comparing given and predicted
					if ~(h==y(1, image_no))
                        %disp('error');
                        no_of_ones(1, image_no) = 1;
                    end
                end
                
                error = w * no_of_ones';
   
				%error more than random guess/chance, flip threshold
                if error > 0.5
                    s = -1 * s;
                    error = w * (~no_of_ones)';
                    %error = 1-error;
                else
                    %disp('Not Enter?')
                end
				
				%error less than minimum error
                if error < min_error
                    %disp(strcat('error reduced ',num2str(error), '    ', num2str(min_error)));
                    min_error = error;
                    optimal_theta = possible_thetas(theta_index);
                    polarity = s;
                end
            end
			
            tmp = zeros(1, 3);
            tmp(1,1) = optimal_theta;
            tmp(1,2) = polarity;
			
			%contains optimal_theta, polarity
            wclassifier(w_i, :) = tmp;
            %wclassifier(w_i,1) = optimal_theta;
            %wclassifier(w_i,2) = polarity;
        end
		
        %using computed theta for each weak classifier, computing weighted error for each weak classifier
        %find the classifier with lowest error - obtain it's theta, alpha, polarity
        
        error_arr = zeros(1, size(wclassifier, 1));
		
        root01 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\err_arr';
        cd(root01);

        ff = wclassifier(:,2);
		
		%for each weak classifier, compute error using optimal theta/error
        parfor w_i = 1:size(wclassifier,1)
		%for w_i=1:size(wclassifier,1)
            
			no_of_ones = zeros(1, N);
            
			%iterating over all images
			for image_no = 1:size(features,2)
                
				%face -> 1, non-face -> -1
				if features(w_i, image_no) < wclassifier(w_i,1)
                    h = -1*ff(w_i);
                else
                    h = ff(w_i);
                end
                
				%comparing given and predicted
				if ~(h==y(1, image_no))
                    no_of_ones(1, image_no) = 1;
                end
            end
            error = w*no_of_ones';
            error_arr(1, w_i) = error;
        end
		
        [error_arr, ind] = sort(error_arr);
        if (t==1 || t==11 || t==51 || t==101)
            save(strcat('error_arr_', num2str(t),'.mat'), 'error_arr'); % saving errors at the specified time steps
        end
        
		min_error = error_arr(1,1);
        index_best_feature = ind(1,1);
        optimal_theta = wclassifier(index_best_feature, 1);
        polarity = wclassifier(index_best_feature, 2);
        
        if ~(ismember(index_best_feature, wclassifier_index))
            alpha_best_feature = 0.5*log((1-min_error)/min_error);
            wclassifier(index_best_feature,3) = alpha_best_feature; % setting alpha
        end
        
		wclassifier_index(1,t) = index_best_feature; % best feaure at time step t
        
       root02 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\misc';
       cd(root02);
        
        %saving selected weak classifier, it's index and weights
        if mod(t,10)==1
            save(strcat('wclassifier_index_iter',num2str(t),'.mat'),'wclassifier_index');
            save(strcat('wclassifier_iter',num2str(t),'.mat'), 'wclassifier');
            save(strcat('weight_iter',num2str(t),'.mat'), 'w');
        end
		
        %updating weights of data points based on alpha, weak classifier and y
        for image_no=1:size(features,2)
            if features(index_best_feature, image_no)<wclassifier(index_best_feature,1) % face 1, non face -1
                h = -1*wclassifier(index_best_feature,2);
            else
                h = wclassifier(index_best_feature,2);
            end
            w(1, image_no) = w(1, image_no) * exp(-1*y(1,image_no)*wclassifier(index_best_feature,3)*h);
        end
		%renormalize weights
        w = w/sum(w);
		%fprintf('Completed T=%d in %.2f seconds \n',t,toc);
    end
	
    root03 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\StrongClassifier';
    cd(root03);
    
	index_strongClass = [index_best_feature];
    theta_strongClass = [wclassifier];%for Realboost
    save(strcat('theta_strongClass_',sprintf('%.2f',T-1),'_T_Iterations.mat'),'theta_strongClass');
    save(strcat('index_strongClass_',sprintf('%.2f',T-1),'_T_Iterations.mat'),'index_strongClass');
    %STRONG BEST FEATURE THETA AND FROZEN THETA INDEX
    %save('adaboost.mat','-v7.3','alpha','index','theta','s','y_hat','eps','err','weak_err');
    fprintf('Running AdaBoost %d with features from %d images took %.2f secs.\n', size(filters, 1), N, toc);
else
    %load('adaboost.mat','alpha','index','theta','s','y_hat','eps','err','weak_err');
end


%----------------------------------------------------------------------------------
%%plotting top-20 haar filters
%----------------------------------------------------------------------------------
% implement this

%T- selected weak classifiers and filters(A,B,C,D)


if flag_top_20_haar_filters
    alpha_arr = zeros(1,20);
    
	for f = 1:20
        index = wclassifier_index(1,f);
        alpha_arr(1, f) = wclassifier(index, 3);
        set(0,'defaultfigurecolor',[1 1 1]);
        fig = figure(f);
        zoom on;
        set(gca,'LineWidth',1, 'FontName','cmr12');
        
		if size(filters{index,1},1)>1
            rectangle('Position',filters{index, 1}(1,1:4),'FaceColor','white');
            rectangle('Position',filters{index, 1}(2,1:4),'FaceColor','white');
        else
            rectangle('Position',filters{index, 1},'FaceColor','white');
        end
        
		if  size(filters{index,2},1)>1
            rectangle('Position',filters{index, 2}(1,1:4),'FaceColor','black');
            rectangle('Position',filters{index, 2}(2,1:4),'FaceColor','black');
        else
            rectangle('Position',filters{index, 2},'FaceColor','black');
        end
        
		title(sprintf('alpha = %d',alpha_arr(1, f)));
        axis([0 17 0 16]);
        daspect([1 1 1]);
        
        root1 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\Haar_filters';
        cd(root1);

        saveas(fig, strcat(num2str(f),'.jpeg'));
    end

    %plotting subplot of Haar filters
    set(0,'defaultfigurecolor',[1 1 1]);
    sub = figure(21);
    title('Top 20 Haar Filters after Boosting', 'interpreter','latex','FontSize',15);

    for f = 1:20 
        index = wclassifier_index(1,f);
        alpha_arr(1, f) = wclassifier(index, 3);
        subplot(4,5,f);
        zoom on;
        set(gca,'LineWidth',1, 'FontName','cmr12');
        
		if size(filters{index,1},1)>1
            rectangle('Position',filters{index, 1}(1,1:4),'FaceColor','white');
            rectangle('Position',filters{index, 1}(2,1:4),'FaceColor','white');
        else
            rectangle('Position',filters{index, 1},'FaceColor','white');
        end
        
		if  size(filters{index,2},1)>1
            rectangle('Position',filters{index, 2}(1,1:4),'FaceColor','black');
            rectangle('Position',filters{index, 2}(2,1:4),'FaceColor','black');
        else
            rectangle('Position',filters{index, 2},'FaceColor','black');
        end
        
		%title(sprintf('Filter %d',f));
        title(sprintf('alpha = %d',alpha_arr(1, f)));
        axis([0 17 0 16]);
        daspect([1 1 1]);
    end
    
    saveas(sub,sprintf('Subplot_20HaarFilters'),'png');
    
    %root2 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\Haar_filters';
    %cd(root2);
	save('alphas_top_20.mat','alpha_arr');
end

%----------------------------------------------------------------------------------
%%plotting training error
%----------------------------------------------------------------------------------
%implement this

if flag_plot_train_error
    disp(T);
	
	%creating y labels for data
    y = ones(1, N);
    y(1, N_pos+1:N_pos+N_neg) = -1;
    
	training_error = zeros(1, T);
    classifier_response = zeros(1, size(features,2));
    
	%running for all selected weak classifiers
	for t = 1:T
        index = wclassifier_index(1,t);
        theta = wclassifier(index, 1);
        polarity = wclassifier(index, 2);
        alpha = wclassifier(index, 3);
        h = zeros(1, size(features,2));
        if(polarity == 1)
            h =  double(features(index,:) >= theta);
        else
            h =  double(features(index,:) < theta);
        end
        h(h==0) = -1;

        classifier_response(1, :) = classifier_response(1, :)+ alpha*h;
        incorrect = 0;
        
		for i=1:size(classifier_response,2)
            %compare classfier response with y
            if (classifier_response(1, i)>0 && y(1, i)<0)
                incorrect = incorrect+1;
            elseif (classifier_response(1, i)<0 && y(1, i)>0)
                incorrect = incorrect+1;
            end
        end
        training_error(1,t) = incorrect/size(features,2);
    end
    
	root2 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\Train_Error';
    cd(root2);
    
	%figure properties
	trainError=figure(23);
    set(0,'defaultfigurecolor',[1 1 1]);
    plot(1:T, training_error, 'LineWidth',1.5);%CHANGE to T
    %title('Training error');
    title('Training error of strong classifier over T iterations', 'interpreter','latex','FontSize',15);
    xlabel('Number of iterations, T', 'interpreter','latex','FontSize',15);
    ylabel('Training error of strong classifier', 'interpreter','latex','FontSize',15);
    grid on;
    set(gca,'LineWidth',1, 'FontName','cmr12');
    saveas(trainError,sprintf('TrainError'),'png');
    end

%----------------------------------------------------------------------------------
%%plotting training errors of top-1000 weak classifiers
%----------------------------------------------------------------------------------
%implement this

if flag_plot_train_error_weak_classifier
    required_steps = [1, 11, 51, 101];
    figure;
    hold on;
	
    root3 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\err_arr';
    cd(root3);
    
	for t=1:size(required_steps,2)
        time_step = required_steps(1, t);
        load(strcat('error_arr_',num2str(time_step),'.mat'));
        error_top_1000_weak_classifiers = error_arr(1,1:1000);
        
        if t==1 
            clr='r--';
        end
        if t==2
            clr='b-.';
        end
        if t==3
            clr='k:';
        end
        if t==4
            clr='m';
        end
        
        plot(1:1000, error_top_1000_weak_classifiers, clr, 'LineWidth',1.5); %min 1000 features
 
    end
	
	%figure properties
    set(0,'defaultfigurecolor',[1 1 1]);
    title('Training error for 1000 weak classifiers', 'interpreter','latex','FontSize',15);
    xlabel('Number of weak classifiers', 'interpreter','latex','FontSize',15);
    ylabel('Training error', 'interpreter','latex','FontSize',15);
    l = legend('T=0','T=10', 'T=50', 'T=100');
    set(l, 'interpreter','latex','FontSize',15,'Location', 'northeastoutside');
    grid on;
    set(gca,'LineWidth',1, 'FontName','cmr12');
    saveas(gcf,sprintf('TrainError_1000weakClassifiers'),'png');
    hold off;
  
end

%----------------------------------------------------------------------------------
%%plotting negative positive histograms
%----------------------------------------------------------------------------------
%implement this

if flag_plot_histograms
    y = ones(1, N);
    y(1, N_pos+1:N_pos+N_neg) = -1; % creating y labels for data
    required_steps = [1, 11, 51, 101];
    
	for step=2:size(required_steps, 2)
        figure;
        classifier_response = zeros(1, size(features,2)/2);
        time_step = required_steps(1,step);
        
		%running for all selected weak classifiers
		for t=1:T
            index = wclassifier_index(1,t);
            theta = wclassifier(index, 1);
            polarity = wclassifier(index, 2);
            alpha = wclassifier(index, 3);
            h = zeros(1, size(features,2)/2);
            
			if(polarity == 1)
                h =  double(features(index,1:size(features,2)/2) >= theta);
            else
                h =  double(features(index,1:size(features,2)/2) < theta);
            end
            h(h==0) = -1;
            classifier_response(1, :) = classifier_response(1, :)+ alpha*h;
        end
		
        histo_1 = histogram(classifier_response);
        name = strcat('positive_classifier_response_', num2str(time_step),'.mat');
        save(name,'classifier_response');
            
        hold on;
        classifier_response = zeros(1, size(features,2)/2);
        
		%running for all selected weak classifiers
		for t = 1:T 
            index = wclassifier_index(1,t);
            theta = wclassifier(index, 1);
            polarity = wclassifier(index, 2);
            alpha = wclassifier(index, 3);
            h = zeros(1, size(features,2)/2);
            
			if(polarity == 1)
                h =  double(features(index,size(features,2)/2+1:size(features,2)) >= theta);
            else
                h =  double(features(index,size(features,2)/2+1:size(features,2)) < theta);
            end
            h(h==0) = -1;
            classifier_response(1, :) = classifier_response(1, :)+ alpha*h;
        end
		
        histo_2 = histogram(classifier_response);
        name = strcat('negative_classifier_response_', num2str(time_step),'.mat');
        save(name,'classifier_response');
        
		%figure properties
		set(0,'defaultfigurecolor',[1 1 1]);
        title(strcat('Histogram of + and - populations over Strong Classifier for T = ',num2str(required_steps(step)-1)), 'interpreter','latex','FontSize',15);
        l = legend('Positive (face)', 'Negative (non-face)');
        set(l, 'interpreter','latex','FontSize',15, 'Location', 'northeastoutside');
        grid on;
        set(gca,'LineWidth',1, 'FontName','cmr12');
        saveas(gcf,sprintf('histStrongClassifier'),'png');
    end
    hold off;
end

%----------------------------------------------------------------------------------
%%plotting receiver operating characteristics ROC curves
%----------------------------------------------------------------------------------
% implement this

if flag_roc_curve
    figure;
    required_steps = [1, 11, 51, 101];
    %required_steps = [1, 11, 51, 101, 201];
    
	for step=2:size(required_steps, 2)
        time_step = required_steps(1, step);
        positive = load(strcat('positive_classifier_response_', num2str(time_step),'.mat'));
        negative = load(strcat('negative_classifier_response_', num2str(time_step),'.mat'));
        [np, edgesp, binp]  = histcounts(positive.classifier_response);
        [nn, edgesn, binn] = histcounts(negative.classifier_response);
        
        nu1 = mean(binp);
        nu2 = mean(binn);
        sig1 = std(binp);
        sig2 = std(binn);
        
        t=linspace(20,-10,31);
        pfunc= @(x) exp((((x-nu1)/sig1).^2)*(-1/2))*(1/(sig1*sqrt(2*pi)));
        nfunc= @(x) exp((((x-nu2)/sig2).^2)*(-1/2))*(1/(sig2*sqrt(2*pi)));
        
        ind=1;
        for i=t
            trueneg(ind)=quad(nfunc,i,20);
            falsepos(ind)=quad(nfunc,-10,i);
            falseneg(ind)=quad(pfunc,i,20);
            truepos(ind)=quad(pfunc,-10,i);
            ind=ind+1;
        end
       
        tpr=truepos./(truepos+falseneg);
        fpr=falsepos./(falsepos+trueneg);
         
        %if t==1 
        %    clr='r--';
        %end
        if step==2
            clr='b-.';
        end
        if step==3
            clr='k:';
        end
        if step==4
            clr='m';
        end
%        if step==5
%            clr='r--';
%       end
        
        plot(tpr,fpr, clr, 'LineWidth',1.5); %SEEEEEEEEEEEEEEEEEEEE
        %plot(fpr,tpr);
        hold on;
    end
	
	%figure properties
    set(0,'defaultfigurecolor',[1 1 1]);
    title('Receiver Operating Characteristics Curve', 'interpreter','latex','FontSize',15);
    l = legend('T=10','T=50', 'T=100');
    set(l, 'interpreter','latex','FontSize',15, 'Location', 'northeastoutside');
    grid on;
    set(gca,'LineWidth',1, 'FontName','cmr12');
    saveas(gcf,sprintf('ROCcurve'),'png');
    grid on;
end

%----------------------------------------------------------------------------------
%%detecting test faces
%----------------------------------------------------------------------------------
% implement this

root1='C:\Users\VishankBhatia\Desktop\Project2\Testing_Images\Testing_Images';
cd(root1);

flag_neg_hard = 1;

if(flag_neg_hard)
    name = 'Non_face_1.jpg';    
    im = imread(name);
else
    name = 'Face_1.jpg';
    im = imread(name);
end

img = rgb2gray(im);

min_thresh =12; % This is the threshold below which we consizder "no face" scenario b/w 5 to 13
Classizfier = [];

root2 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data';
cd(root2);
load('filters.mat');
load('features.mat');

root3 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\misc';
cd(root3);
load('wclassifier_index_iter101.mat');
load('wclassifier_iter101.mat');

for i = 1:sizze(wclassifier_index, 2)
    Classizfier(i).y_hat = wclassifier(wclassifier_index(i), 1);
    Classizfier(i).polarity = wclassifier(wclassifier_index(i), 2);
    Classizfier(i).alpha = wclassifier(wclassifier_index(i), 3);
    Classizfier(i).filter = filters(wclassifier_index(i), :);
end

% delete(gcp('nocreate'));
% myCluster = parcluster('local');
% delete(myCluster.Jobs);
% parpool(4);
window_sizze = 16;
step = 3;
figure(4);
imshow(img);
scales = [0.2 0.4 0.6 1];
% scales = [0.2 0.4];
rectangles=[];
for num = 1:sizze(scales,2)
    count = 1;
    siz = scales(num);
    fprintf('Image Scaling Factor = %.2f\n', siz);
    feature_sizze = sizze(Classizfier, 2);
    nimg = imresizze(img, siz);
    [m, n] = sizze(nimg);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FRONT ROWS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if siz == 0.2
        row_start = floor(m*0.6);
        row_end = m - window_sizze;
    end 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FRONT MIDDLE ROWS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if siz == 0.4
        row_start = floor(m*0.55);
        row_end = ceil((m - window_sizze)*0.62);
    end 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BACK MIDDLE ROWS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if siz == 0.6
        row_start = floor(0.47*m);
        row_end = ceil((m - window_sizze)*0.60);
    end 
root4 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\HardNegatives';
cd(root4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BACK ROWS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if siz == 1
        row_start = floor(m*0.38);
        row_end = ceil(m*0.49);
    end
%     row_start = 1;
%     row_end = 16;

root5 = 'C:\Users\VishankBhatia\Desktop\Project2\fwdrealboostcode';
cd(root5);
     
    for i = row_start : step : row_end - window_sizze + 1
        for j = 1 : step : n - window_sizze + 1
            window = nimg(i:i+window_sizze-1, j:j+window_sizze-1);
            window = reshape(window, 1, 16 ,16);
            integral_w = integralFunc(window);
            res = 0;
            for k = 1 : feature_sizze
                filter = Classizfier(k).filter;
                threshold = Classizfier(k).y_hat;
                polarity = Classizfier(k).polarity;
                alpha = Classizfier(k).alpha;
                temp = compute_features(reshape(integral_w, 1, 17, 17), filter);
                
                if temp >= threshold
                        res = res + polarity*alpha;
                else
                        res = res - polarity*alpha;
                end
            end    
            if(flag_neg_hard && res > min_thresh)
                imwrite(reshape(window, 16, 16), strcat(num2str(count, '%5d'), '.bmp'));
                count = count + 1;
            else
                if(res > min_thresh)
                    rectangles = addRect(rectangles, res, siz, i, j, window_sizze);
                end
            end
        end
    end
%     if(flag_neg_hard == 0)
%         boundBox(rectangles);
%         rectangles = [];
%     end
end

boundBox(rectangles,1);
%saveas(gcf,strcat('./FaceDetection/step-3-',filename));

disp('Done.');
delete(myCluster.Jobs); %removes unwanted running cores
end



%----------------------------------------------------------------------------------
%% filters
%----------------------------------------------------------------------------------

function features = compute_features(II, filters)
features = zeros(size(filters, 1), size(II, 1));
for j = 1:size(filters, 1)
    [rects1, rects2] = filters{j,:};
    features(j,:) = apply_filter(II, rects1, rects2);
end
end

function I = normalize(I)
[N,~,~] = size(I);
for i = 1:N;
    image = I(i,:,:);
    sigma = std(image(:));
    I(i,:,:) = I(i,:,:) / sigma;
end
end

function II = integral(I)
[N,H,W] = size(I);
II = zeros(N,H+1,W+1);
for i = 1:N
    image = squeeze(I(i,:,:));
    II(i,2:H+1,2:W+1) = cumsum(cumsum(double(image), 1), 2);
end
end

function sum = apply_filter(II, rects1, rects2)
sum = 0;
% white rects
for k = 1:size(rects1,1)
    r1 = rects1(k,:);
    w = r1(3);
    h = r1(4);
    sum = sum + sum_rect(II, [0, 0], r1) / (w * h * 255);
end
% black rects
for k = 1:size(rects2,1)
    r2 = rects2(k,:);
    w = r2(3);
    h = r2(4);
    sum = sum - sum_rect(II, [0, 0], r2) / (w * h * 255);
end
end

function result = sum_rect(II, offset, rect)
x_off = offset(1);
y_off = offset(2);

x = rect(1);
y = rect(2);
w = rect(3);
h = rect(4);

a1 = II(:, y_off + y + h, x_off + x + w);
a2 = II(:, y_off + y + h, x_off + x);
a3 = II(:, y_off + y,     x_off + x + w);
a4 = II(:, y_off + y,     x_off + x);

result = a1 - a2 - a3 + a4;
end

function rects = filters_A()
count = 1;
w_min = 4;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:2:w_max
    for h = h_min:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/2;
                r1_h = h;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x + r1_w;
                r2_y = r1_y;
                r2_w = w/2;
                r2_h = h;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                rects{count, 1} = r1; % white
                rects{count, 2} = r2; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_B()
count = 1;
w_min = 4;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:w_max
    for h = h_min:2:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w;
                r1_h = h/2;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x;
                r2_y = r1_y + r1_h;
                r2_w = w;
                r2_h = h/2;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                rects{count, 1} = r2; % white
                rects{count, 2} = r1; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_C()
count = 1;
w_min = 6;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:3:w_max
    for h = h_min:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/3;
                r1_h = h;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x + r1_w;
                r2_y = r1_y;
                r2_w = w/3;
                r2_h = h;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                r3_x = r1_x + r1_w + r2_w;
                r3_y = r1_y;
                r3_w = w/3;
                r3_h = h;
                r3 = [r3_x, r3_y, r3_w, r3_h];
                
                rects{count, 1} = [r1; r3]; % white
                rects{count, 2} = r2; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_D()
count = 1;
w_min = 6;
h_min = 6;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:2:w_max
    for h = h_min:2:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/2;
                r1_h = h/2;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x+r1_w;
                r2_y = r1_y;
                r2_w = w/2;
                r2_h = h/2;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                r3_x = x;
                r3_y = r1_y+r1_h;
                r3_w = w/2;
                r3_h = h/2;
                r3 = [r3_x, r3_y, r3_w, r3_h];
                
                r4_x = r1_x+r1_w;
                r4_y = r1_y+r2_h;
                r4_w = w/2;
                r4_h = h/2;
                r4 = [r4_x, r4_y, r4_w, r4_h];
                
                rects{count, 1} = [r2; r3]; % white
                rects{count, 2} = [r1; r4]; % black
                count = count + 1;
            end
        end
    end
end
end

function test_sum_rect()
% 1
I = zeros(1,16,16);
I(1,2:4,2:4) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [2, 2, 3, 3]) == 9);
assert(sum_rect(II, [0, 0], [10, 10, 2, 2]) == 0);

% 2
I = zeros(1,16,16);
I(1,10:16,10:16) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [10, 10, 2, 2]) == 4);

% 3
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 1;
I(1,3:6,11:14) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [11, 3, 6, 6]) == 16);

% 4
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 1;
I(1,3:6,11:14) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [3, 4, 4, 4]) == 12);
assert(sum_rect(II, [0, 0], [7, 4, 4, 4]) == 0);
assert(sum_rect(II, [0, 0], [11, 4, 4, 4]) == 12);
assert(sum_rect(II, [0, 0], [3, 3, 4, 4]) == 16);
assert(sum_rect(II, [0, 0], [11, 3, 4, 4]) == 16);

end

function test_filters()

% A
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,5:8,5:8) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_A();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*2))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [r1s, r2s];
    end
end
assert(max_sum == 1);
assert(max_size == 4*4*2);
assert(isequal(min_f, [1 5 4 4 5 5 4 4]));

% B
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,2:5,2:5) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_B();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*2))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [r1s, r2s];
    end
end
assert(max_sum == 1);
assert(max_size == 4*4*2);
assert(isequal(min_f, [2 6 4 4 2 2 4 4]));

% C
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 255;
I(1,3:6,11:14) = 255;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_C();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r1s(2,3) * r1s(2,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*3))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [reshape(r1s', [1,8]), r2s];
    end
end
assert(max_sum == 2);
assert(max_size == 4*4*3);
assert(isequal(min_f, [3 3 4 4 11 3 4 4 7 3 4 4]));

% D
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,2:5,2:5) = 0;
I(1,6:9,6:9) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_D();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r1s(2,3) * r1s(2,4) + r2s(1,3) * r2s(1,4) + r2s(2,3) * r2s(2,4);
    if(and(f_sum > max_sum, f_size == 4*4*4))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [reshape(r1s', [1,8]), reshape(r2s', [1,8])];
    end
end
assert(max_sum == 2);
assert(max_size == 4*4*4);
assert(isequal(min_f, [6 2 4 4 2 6 4 4 2 2 4 4 6 6 4 4]));
end
