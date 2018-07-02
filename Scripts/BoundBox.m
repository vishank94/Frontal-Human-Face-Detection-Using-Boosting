function [] = boundBox(rectangles,ifAdaBoost)
    if(ifAdaBoost)
        green = 12.5;
        yellow = 12; %increase this values, boxes of that colour will decrease.
        red =9;
        %green = 12.5;
        %yellow = 12; %increase this values, boxes of that colour will decrease.
        %red =9;
    else%else is for hard negative mining
        green = 1.25;
        yellow = 1.0;
        red = 0.5;
    end  
    for i = 1: size(rectangles, 1)
        num = rectangles(i,1);
        x_coord = rectangles(i,2);
        y_coord = rectangles(i,3);
        osize = rectangles(i,4);
        if(num >= green)
            rectangle('Position', [x_coord, y_coord, osize, osize], 'EdgeColor', 'g');
        elseif(num >= yellow)
            rectangle('Position', [x_coord, y_coord, osize, osize], 'EdgeColor', 'y');
        elseif(num >= red)
            rectangle('Position', [x_coord, y_coord, osize, osize], 'EdgeColor', 'r');
        end
    end
end