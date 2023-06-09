SEARCH_FOLDER="/home/jacklishufan/xView3_second_place/cluster/exps"
for f in $(find $SEARCH_FOLDER -name '*.sh');
do (echo "launching  $f"; $1 $f ; echo "Done")
done