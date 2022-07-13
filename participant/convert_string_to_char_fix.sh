#!/bin/bash 


echo "Converting strings to char for each scenes.mat!"

for d in *p*/ ; do
    echo $d;
    cd $d;
    ~/local/MATLAB/R2021b/bin/matlab -nodisplay -batch "load('scenes.mat'); var=controllib.internal.util.hString2Char(var); save('scenes_fixed.mat')";
    cd ..;
done

echo "Done!"

# make sure to do the following to run the script from terminal
# chmod +x convert_string_to_char_fix.sh


