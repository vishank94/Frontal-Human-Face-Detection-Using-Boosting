%Load the clasifier model from adaboost:
flag_boosting = 1;
flag_histograms = 1;

if(flag_boosting == 1)
   
    
flag_boosting=1;
flag_plot_histogram=1;
N_pos=5000; N_neg=5000;
N=N_pos+N_neg;

root1 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data';
cd(root1);
load('filters.mat');
load('features.mat');

root2 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\misc';
cd(root2);
load('wclassifier_index_iter101.mat');
load('wclassifier_iter101.mat');

tic;
        for i = 1:size(wclassifier_index, 2)
            Classifier(i).y_hat = wclassifier(wclassifier_index(i), 1);
            Classifier(i).polarity = wclassifier(wclassifier_index(i), 2);
            Classifier(i).alpha = wclassifier(wclassifier_index(i), 3);
            Classifier(i).filter = filters(wclassifier_index(i), :);
            Classifier(i).feature = features(wclassifier_index(i), :);
        end
        fprintf('Loaded model. Took %.2f sec',toc);
    end

    face = 5000;
    nonface = 5000;
    label = [ones(face, 1); -1*ones(nonface, 1)];
    weight = [ones(face, 1) / (face*2); ones(nonface, 1) / (nonface*2)];
    epslion = 1e-5;

    feature_num = size(Classifier, 2);
    Feature = [];
    temp_min = 3000;
    temp_max = -3000;
    for i = 1 : feature_num
        Feature(i).feature = Classifier(i).feature;
        if(min(Feature(i).feature)<temp_min)
            temp_min = min(Feature(i).feature);
        end
        if(max(Feature(i).feature)>temp_max)
            temp_max = max(Feature(i).feature);
        end    
    end
    % feature_num = size(Feature, 2);
    % feature_num = 1;
    RealClassifier = [];

    scale = 50;
%%
    T = feature_num;
    for i = 1 : T
        tic;
        Z = zeros(feature_num, 1);
        Htb = zeros(scale+2, feature_num); 
    %     Spaces = zeros(scale+1, feature_num);

        pos_weight = weight(label == 1);
        neg_weight = weight(label == -1);

    %     b_min = -3000;
    %     b_max = 3000;
        b_min = temp_min;
        b_max = temp_max;

        spaces = linspace(b_min, b_max, scale+1);
        for j = 1 : feature_num
            %output = classifyByFeature(dataset, Classifier(j).filter);
            output = Feature(j).feature;
            pos_res = output(label == 1);
            neg_res = output(label == -1);

            htb = zeros(scale+2, 1);
            qt = zeros(scale+2, 1);
            pt = zeros(scale+2, 1);

            sump = 0;
            sumq = 0;

            for k = 1 : size(spaces, 2)
                if k == 1
                    pt(k) = sum(pos_weight(find(pos_res <= spaces(k))));
                    qt(k) = sum(neg_weight(find(neg_res <= spaces(k))));
                else
                    pt(k) = sum(pos_weight(find(pos_res > spaces(k-1) & pos_res <= spaces(k))));
                    qt(k) = sum(neg_weight(find(neg_res > spaces(k-1) & neg_res <= spaces(k))));
                end
            end
            pt(end) = sum(pos_weight(find(pos_res > spaces(end))));
            qt(end) = sum(neg_weight(find(neg_res > spaces(end))));

            htb = log((pt + epslion) ./ (qt + epslion)) / 2;
            Htb(:, j) = htb;
    %         Spaces(:, j) = spaces;
            z = 2 * sum(sqrt(pt.*qt));
            Z(j) = z;
        end

        [z_min, ind] = min(Z);
        RealClassifier(i).feature = Feature(ind).feature;
        RealClassifier(i).htb = Htb(:, ind);
        RealClassifier(i).filter = Classifier(ind).filter;
        htb = Htb(:, ind);

        %output = classifyByFeature(dataset, Classifier(ind).filter);
        output = Feature(ind).feature;
        pos_res = output(label == 1);
        neg_res = output(label == -1);
        pos_weight = weight(label == 1);
        neg_weight = weight(label == -1);

        for k = 1 : size(htb, 1)
            if k == 1
                pos_weight(find(pos_res <= spaces(k))) = pos_weight(find(pos_res <= spaces(k)))*exp(-htb(k));
                neg_weight(find(neg_res <= spaces(k))) = neg_weight(find(neg_res <= spaces(k)))*exp(htb(k));
            elseif k == size(htb, 1)
                pos_weight(find(pos_res > spaces(k-1))) = pos_weight(find(pos_res > spaces(k-1)))*exp(-htb(k));
                neg_weight(find(neg_res > spaces(k-1))) = neg_weight(find(neg_res > spaces(k-1)))*exp(htb(k));
            else
                pos_weight(find(pos_res > spaces(k-1) & pos_res <= spaces(k))) = pos_weight(find(pos_res > spaces(k-1) & pos_res <= spaces(k)))*exp(-htb(k));
                neg_weight(find(neg_res > spaces(k-1) & neg_res <= spaces(k))) = neg_weight(find(neg_res > spaces(k-1) & neg_res <= spaces(k)))*exp(htb(k));
            end
        end

        weight = [pos_weight; neg_weight];
        norm = sum(weight);
        weight = weight / norm;
        fprintf('Completed T=%d took %.2f secs.\n', i, toc);
    end
    root3 = 'C:\Users\VishankBhatia\Desktop\Project2\project2_code_and_data\Realboost';
    cd(root3);
    save('RealClassifier.mat', 'RealClassifier');
    save('partitions.mat','spaces');

% else
%     tic;
%     load('RealClassifier.mat', 'RealClassifier');
%     load('partitions.mat','spaces');
%     fprintf('Loaded Real Classifier and Partitions. Took %.2f secs.\n', toc);
%end
%%

if(flag_histograms == 1)    
    required_steps = [1,10,50,100];
    for step = required_steps
        figure();
        %Positive Histogram
        classifier_response = zeros(1, size(RealClassifier(step).feature,2)/2);
        for t=1:step % for all 20 classifiers %% CHANGE TO T
            feature = RealClassifier(t).feature;
            h = zeros(1, size(feature,2)/2);
            for i = 1: (size(feature, 2)/2)
                id = size(find(spaces <= feature(:, i)), 2);
                h(1,i) = RealClassifier(t).htb(id+1);
            end
            classifier_response(1, :) = classifier_response(1, :)+ h;
        end
        histo_1 = histogram(classifier_response);
        name = strcat('positive_classifier_response_', num2str(step),'.mat');
        save(name,'classifier_response');
        hold on;
        %Negative Histogram
        classifier_response = zeros(1, size(feature,2)/2);
        for t=1:step % for all 20 classifiers %% CHANGE TO T
            feature = RealClassifier(t).feature;
            h = zeros(1, size(feature,2)/2);
            for i = size(feature, 2)/2+1: size(feature, 2)
                id = size(find(spaces <= feature(:, i)), 2);
                h(1,i-size(feature, 2)/2) = RealClassifier(t).htb(id+1);
            end
            classifier_response(1, :) = classifier_response(1, :)+ h;
        end
        histo_2 = histogram(classifier_response);
        name = strcat('negative_classifier_response_', num2str(step),'.mat');
        save(name,'classifier_response');
        %hist2 = histogram(feature(:,size(feature, 2)/2+1:size(feature, 2)),spaces);
        legend('Positive', 'Negative');
       % saveas(gcf,['./RealboostPlots/Histograms/hist-' num2str(step) '.jpg']);
    end
end    
%%
if 1
    figure;
    required_steps = [1, 10, 50, 100];
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
            %clr = first;
        %end
        if step==2
            clr='b-.';
            %clr = second;
        end
        if step==3
            clr='k:';
            %clr = third;
        end
        if step==4
            clr='m';
            %clr = fourth;    
        end

        
        plot(fpr,tpr, clr, 'LineWidth',1.5);

      
                
        hold on;
    end
end