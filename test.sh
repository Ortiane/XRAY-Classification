python test.py \
  --image_dir=../data/preprocessing \
  --csv_file=../data/Data_Entry_2017.csv \
  --epochs=30 \
  --train_txt=../data/train_val_list.txt \
  --test_txt=../data/test_list.txt \
  --batch_size=32 \
  --checkpoint_path=save/model.h5 \
  --model_name=efficientnet
