# Replaces the xids within a Jupyter notebook. Note: The search file is specified in the code, defaulting to results_plot.ipynb
# INPUT: The find/replace map in the form '[FIND] --> [REPLACE]\n'.
# OUTPUT: The altered file.

INPUT_FILE="results_plot.ipynb"
OUTPUT_FILE="results_plot_new.ipynb"

# Program start
echo "Please enter the query in the form: '[FIND] --> [REPLACE]\n'"
cp $INPUT_FILE $OUTPUT_FILE

# Reads in the find/replace input
while read line
do
    replace_string=("${replace_string[@]}" $line)
done

# Conducts the find and replacement for each pair
state="find"
for elem in ${replace_string[@]}
do
    if [ "$state" = "find" ]; then
	key=$elem
	state="transition"
    elif [ "$state" = "transition" ]; then
	state="replace"
    elif [ "$state" = "replace" ]; then
	value=$elem
	echo "$key --> $value"

	sed -i "s/\(xids *= *\[\(.*, *\)*\)${key}\(.*\]\)/\1${value}\3/g" $OUTPUT_FILE
	sed -i "s/\(xid *= *\)${key}/\1${value}/g" $OUTPUT_FILE

	state="find"
    fi
done
