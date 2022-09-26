# graph analysis with network x

dependencies: pandas numpy matplotlib libpysal networkx seaborn scipy

# how to run:

python nwx_analyze.py -h

## example
### NOTE: you now need to specify a path to where you keep your images!!!
python nwx_analyze.py -i ~/Documents/Ovarial_22/ML_TMA1/obj_class_TMA1.csv -p CD45:PANCK --tiff_dir ~/Documents/Ovarial_22/imj_out_1

# Notes

## Regarding class1:class2
while developing the script, class1 was assumed to be immune cells and class2 cancer cells, this means for example that group degree centrality is calculated for whatever is on the left side!
