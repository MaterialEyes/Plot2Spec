
DATASET_DIR=/datashare/datasets_3rd_party
DATASETS=tusimple culane bdd
DATASET=tusimple

TUSIMPLE_DATA_DIR=$(DATASET_DIR)/tusimple-benchmark
CULANE_DATA_DIR=$(DATASET_DIR)/CULane
BDD_DATA_DIR=$(DATASET_DIR)/bdd/bdd100k

ifeq ($(DATASET), tusimple)
	DATA_DIR=$(TUSIMPLE_DATA_DIR)
	TEST_FILE=$(DATA_DIR)/test_tasks_0627.json
	THICKNESS=5
	IMG_WIDTH=512
	IMG_HEIGHT=256
else ifeq ($(DATASET), culane)
	DATA_DIR=$(CULANE_DATA_DIR)
	TEST_FILE=$(META_DIR)/$(DATASET).json
	THICKNESS=12
	IMG_WIDTH=800
	IMG_HEIGHT=288
else ifeq ($(DATASET), bdd)
	DATA_DIR=$(BDD_DATA_DIR)
	TEST_FILE=$(META_DIR)/$(DATASET).json
	THICKNESS=8
	IMG_WIDTH=800
	IMG_HEIGHT=288
else
    @echo 'Unknown $(DATASET)!!!'
endif

OUT_DIR=/datashare/users/sang/works/lanenet/output
META_DIR=$(OUT_DIR)/metadata
MODEL_DIR=$(OUT_DIR)/model
TEST_DIR=$(OUT_DIR)/test

# Variables
GT_FILE=$(DATA_DIR)/test_label.json

BATCH_SIZE?=16
TEST_BATCH_SIZE?=16
LEARNING_RATE?=0.0001
CNN_TYPE?=unet

EXP_NAME=$(DATASET)_$(CNN_TYPE)_b$(BATCH_SIZE)_lr$(LEARNING_RATE)
MODEL_FILE=$(MODEL_DIR)/$(EXP_NAME).pth
PRED_FILE=$(MODEL_FILE:.pth=_predictions.json)
TRAIN_LOG_FILE=$(MODEL_FILE:.pth=_training.log)
TEST_OUT_DIR=$(TEST_DIR)/$(EXP_NAME)

SPLITS=train val test

metadata: $(patsubst %,$(META_DIR)/%.json,$(DATASETS))
$(META_DIR)/tusimple.json:
	python src/metadata.py --input_dir $(TUSIMPLE_DATA_DIR) \
		--dataset tusimple \
		--output_file $@

$(META_DIR)/culane.json:
	python src/metadata.py --input_dir $(CULANE_DATA_DIR) \
		--dataset culane \
		--output_file $@

$(META_DIR)/bdd.json:
	python src/metadata.py --input_dir $(BDD_DATA_DIR) \
		--dataset bdd \
		--output_file $@

# Generate binary segmentation image & instance segmentation images from
# the annotation data
BIN_DIR=$(OUT_DIR)/bin_images # contains binary segmentation images
INS_DIR=$(OUT_DIR)/ins_images # contains instance segmentation images
generate_label_images:
	python src/gen_seg_images.py $(META_DIR)/tusimple.json $(DATA_DIR) \
		--bin_dir $(BIN_DIR) \
		--ins_dir $(INS_DIR) \
		--splits train val \
		--thickness $(THICKNESS)

#START_FROM=$(MODEL_DIR)/$(DATASET)_current.pth
train: $(MODEL_FILE)
$(MODEL_FILE): $(META_DIR)/$(DATASET).json 
	python src/train.py $^ $@ \
		--image_dir $(DATA_DIR) \
		--batch_size $(BATCH_SIZE) \
		--num_workers 8 \
		--cnn_type $(CNN_TYPE) \
		--embed_dim 4 \
		--dataset $(DATASET) \
		--width $(IMG_WIDTH) \
		--height $(IMG_HEIGHT) \
		--thickness $(THICKNESS) \
		2>&1 | tee $(TRAIN_LOG_FILE)

test: $(PRED_FILE)
$(PRED_FILE): $(MODEL_FILE) $(TEST_FILE) 
	python src/test.py $< \
		--output_file $@ \
		--meta_file $(word 2, $^) \
		--image_dir $(DATA_DIR) \
		--output_dir $(TEST_OUT_DIR) \
		--loader_type $(DATASET)test \
		--num_workers 8 \
		--batch_size $(TEST_BATCH_SIZE)

# The provided evaluation script was written in Python 2, while this project use Python 3
# Solution is to use an Python 2 env for the evaluation
# Makefile uses /bin/sh as the default shell, which does not implement source
# change SHELL to /bin/bash to activate the Python 2 environment
SHELL=/bin/bash 
eval_tusimple: $(PRED_FILE) $(GT_FILE) 
	source activate py2 && \
		python tusimple-benchmark/evaluate/lane.py $^ && \
	source deactivate


# Show the results for each image on the test set (by turning on the show_demo switch)
demo_tusimple: $(MODEL_FILE) $(META_DIR)/tusimple.json
	python src/test.py $< \
		--meta_file $(word 2, $^) \
		--image_dir $(DATA_DIR) \
		--output_dir $(OUT_DIR)/demo_tusimple \
		--loader_type meta \
		--batch_size 1 --show_demo

# Examples of make rules to test lane detection from an image directory
TEST_IMG_DIR?=/datashare/datasets_3rd_party/bdd/bdd100k/images/100k/test
test_bdd: $(MODEL_FILE) 
	python src/test.py $^ \
		--image_dir $(TEST_IMG_DIR) \
		--output_dir $(TEST_OUT_DIR)/bdd_test \
		--loader_type dirloader \
		--image_ext jpg \
		--batch_size 1 

TEST_IMG_DIR?=/datashare/datasets_ascent/extracted/2018-12-13/13-15-33/part_70/sync/low_res_rear_cam
test_20181213_rear: $(MODEL_FILE) 
	python src/test.py $^ \
		--image_dir $(TEST_IMG_DIR) \
		--output_dir $(TEST_OUT_DIR)/20181213_low_res_rear_cam \
		--loader_type dirloader \
		--image_ext png \
		--batch_size 1 
	ls -t $(TEST_IMG_DIR)/*.png | xargs cat | /usr/bin/ffmpeg -framerate 25 -i - -r 25 -c:v libx264 -pix_fmt yuv420p $(TEST_OUT_DIR)/20181213_low_res_rear_cam.mp4 


TEST_IMG_DIR?=/datashare/datasets_ascent/extracted/2018-12-13/13-15-33/part_70/sync/low_res_front_left_cam
test_20181213_frontleft: $(MODEL_FILE) 
	python src/test.py $^ \
		--image_dir $(TEST_IMG_DIR) \
		--output_dir $(TEST_OUT_DIR)/20181213_low_res_front_left_cam \
		--loader_type dirloader \
		--image_ext png \
		--batch_size 1 
	cat $(TEST_IMG_DIR)/*.png | ffmpeg -framerate 25 -i - -r 25 -pix_fmt yuv420p $(TEST_OUT_DIR)/20181213_low_res_front_left_cam.mp4 

TEST_IMG_DIR?=/datashare/datasets_ascent/extracted/2018-12-13/13-15-33/part_70/sync/low_res_front_right_cam
test_20181213_frontright: $(MODEL_FILE) 
	python src/test.py $^ \
		--image_dir $(TEST_IMG_DIR) \
		--output_dir $(TEST_OUT_DIR)/20181213_low_res_front_right_cam \
		--loader_type dirloader \
		--image_ext png \
		--batch_size 1 
	cat $(TEST_IMG_DIR)/*.png | ffmpeg -framerate 25 -i - -r 25 -pix_fmt yuv420p $(TEST_OUT_DIR)/20181213_low_res_front_right_cam.mp4 

# /datashare/datasets_ascent/cardump/output/2018-11-07-extraction-for-scalabel/sample_compress_output \
	
test_culane_sample: $(MODEL_FILE) 
	python src/test.py $^ \
		--image_dir /datashare/datasets_3rd_party/CULane/driver_100_30frame/05251517_0433.MP4 \
		--output_dir $(OUT_DIR)/culane_test_sample \
		--loader_type dirloader \
		--image_ext jpg \
		--batch_size 1 

test_video:
	python src/test_video.py $(MODEL_FILE) \
		--input_file /home/sang/clones/Advanced-Lane-Lines/harder_challenge_video.mp4 \
		--output_file harder_challenge_video_lanenet.mp4 \
		--genline_method maxprob

test_ascent_video:
	python src/test_video.py $(MODEL_FILE) \
		--input_file /datashare/users/sang/datasets/video.mp4 \
		--output_file video.mp4 \
		--genline_method maxprob
