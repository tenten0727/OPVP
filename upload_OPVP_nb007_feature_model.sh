cp /home/yoshikawa/work/kaggle/OPVP/dataset-metadata.json "$1"

kaggle datasets version -p "$1" -m 'update_feature_model'