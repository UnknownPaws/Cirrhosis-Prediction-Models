# Original CSV from kaggle
# https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction
egrep -v "NA|CL" cirrhosis.csv > temp.csv
cut -d, -f2-19 temp.csv > cirrhosis.csv
rm temp.csv
