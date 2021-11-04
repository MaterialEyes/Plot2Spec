CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder test_demo_images/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth