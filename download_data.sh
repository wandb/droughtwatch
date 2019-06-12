DEST=${VARIABLE:-"data"}
if [ ! -d "$DEST" ]; then
  echo "Destination doesn't exist, creating."
  mkdir $DEST
fi
SOURCE=gs://satellite_processed_pipeline/output4/*
echo "Downloading data from " $SOURCE " to " $DEST
gsutil -m cp -r $SOURCE $DEST
