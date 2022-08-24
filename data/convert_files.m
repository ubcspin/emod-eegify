files = dir('p*/*.mat');

for file = files'
    [filepath, name, ext] = fileparts(file.name);
    filename = [file.folder '/' file.name];
    
    if contains(name, 'events')
        temp = load(filename);
        temp = temp.var;
        writetable(temp, [file.folder '/' name '.csv'])

    else
        try
            try
                temp = load(filename);
            catch
                temp = load(filename, '-ascii');
            end
            
            while length(temp) == 1
                temp = temp.var;
            end
    
            try
                writetable(temp, [file.folder '/' name '.csv']);
            catch
                write(array2table(temp), [file.folder '/' name '.csv']);
            end
        catch
            disp(filename);
        end
    end
end