cp /home/yoshikawa/work/kaggle/OPVP/dataset-metadata-stock31.json "$1"
mv "$1"/dataset-metadata-stock31.json "$1"/dataset-metadata.json
kaggle datasets version -p "$1" -m 'update_feature_model'