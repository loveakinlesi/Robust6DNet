python datagen/find_high_visib_images.py --obj_id 2
python datagen/generate_annotations.py --obj_id 2
python datagen/generate_test_annotations.py --obj_id 2
python datagen/generate_crops.py --obj_id 2
python datagen/generate_heatmaps.py --obj_id 2
python training/train_lm_r6dnet.py --obj_id 2

echo "complete"