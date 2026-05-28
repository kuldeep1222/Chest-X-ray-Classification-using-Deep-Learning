# EXPERIMENT--1

Model: Simple 3-layer CNN from scratch
Input: 128x128 grayscale, batch=32, learning rate(lr)=0.001, epochs=10
Result: 96.88% test accuracy
Observation: 1.Loss was noisy (In epoch 8 spiked to 5.3143 from 5.1798) and but model converged well 
	         2.Wrong prediction on images which are not in dataset

# EXPERIMENT--2
Model: MobileNetV2 pretrained on ImageNet, features frozen
Input: 224x224 pseudo-RGB, batch=32, learning rate(lr)=0.001, epochs=10
Result: 91.88% test accuracy

Observation: Accuracy Surprisingly worse than Experiment 1. Three reasons:
1. Full freezing limited adaptability to X-ray domain
2. Domain gap — ImageNet features don't transfer cleanly to grayscale medical images
3. lr=0.001 too aggressive for fine-tuning pretrained weights
4. Predicts well on unseen images
5. 
Next steps: Unfreeze last few blocks, reduce lr to 0.0001,
add data augmentation
