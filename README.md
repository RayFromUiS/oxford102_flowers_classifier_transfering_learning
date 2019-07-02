# oxford102_flowers_classifier

This project is aiming to predict the Oxford 102 flowers datasets.

Process is carried with such steps:
  1. preprocess the image data and save inside of dataloader
  2. import the pretrained model architecture and define fully connect layer,optimizer function
  3. feed train dataloader and validation dataloader to predefined model.
  4. test on test datasets
  5. saving the checkpoint for further use
  6. load at cpu for image recognization
  7. preprocess data image is recognized by model and predict 5 top possiblities.
