clear
[file,path] = uigetfile('*.xml');
if file == 0 
    disp('No file selected.')
elseif path == 0
    disp('No path selected.')
else
    xDoc = xmlread(strcat(path,file));
    xRoot= xDoc.getDocumentElement();
    xDepths = xRoot.getElementsByTagName('depth');
    l = xDepths.getLength();
    if xDepths.getLength() > 0
        xDepth = xDepths.item(0);
        rows = xDepth.getElementsByTagName('rows');
        cols = xDepth.getElementsByTagName('cols');
        datas = xDepth.getElementsByTagName('data');
        if rows.getLength()*cols.getLength()*datas.getLength() == 1
            row = str2num(rows.item(0).getTextContent());
            col = str2num(cols.item(0).getTextContent());
            data = datas.item(0).getTextContent();
            data_cell = strsplit(char(data));
            xyzPoints = cell(row*col,3);
            for i = 1:row
               for j = 1:col
                   index = (i-1)*col + j;
                   xyzPoints{index,3} = row-i+1;
                   xyzPoints{index,1} = j;
                   xyzPoints{index,2} = str2double(data_cell{index+1});
               end
            end
            ptCloud = pointCloud(cell2mat(xyzPoints));
            pcshow(ptCloud)
        else
            disp('No rows/cols/data Element.')
        end
    else
        disp('No depth Element.')
    end
end