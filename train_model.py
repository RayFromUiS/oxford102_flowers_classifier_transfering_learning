from torchvision import datasets, transforms, models
import torch
import torch.nn as nn

import time
from torch import optim
import copy
import matplotlib.pyplot as plt
from PIL import Image

def load_pre_trained_model(model,hidden_units):
    
    ''' return the pretraine model,resenet101 or vgg16
    input, 
        model, passing in pretrained model from torch,vision.models
        hidden_units, hidden nodes number for the hidden layer
    
    return,
        model, fully connected layer for the model
        
    '''
    #Preserve the features

    if model == 'resnet101':
        
        model = models.resnet101(pretrained = True)    
        
        for param in model.parameters():
            param.requires_grad = False

            # model.fc.in_features == 2048,must be preserved
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        
    elif model == 'vgg16':
        
        model = models.resnet101(pretrained = True)    
        
        for param in model.parameters():
            param.requires_grad = False
        #in_features nodes must be preserved as well
        model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    return model,optimizer
        
        

def nn_classifer_train_valid(epochs,pre_trained_model,optimizer,train_loader,valid_loader,gpu):   
    
    ''' training and validating the model on the corresponding model and return the classifer
    
    input, epoches,number of training epochs;
           pre_trained_model,import pretrained model from the torch.vision with freezed feature parameters
           criterion, metrics to evaluation the error
           optimizer,optimization function to reducing the error,
           train_datasets,train data from files
           valid_datasets, validation datasets
    outpu, model(image classifier trained from pretrained model)
    '''    
    if gpu:
        device = 'cuda' 
    else:
        device = 'cpu'
    model = pre_trained_model
    model.to(device)
    start = time.time()
    training_loss = 0
    steps = 0
    best_accuracy = 0
    print_every = 50
    val_accuracies= []
    best_model_wts = copy.deepcopy(model.state_dict())
    criterion = nn.NLLLoss()
    for epoch in range(epochs):

        for images,targets in train_loader:

            model = model.train()
            steps  += 1
            images,targets = images.to(device),targets.to(device)        
            #forward propagation
            optimizer.zero_grad()
            log_ps = model.forward(images) #logsoftmaxt
            loss = criterion(log_ps,targets)
            #back propagation
            loss.backward()
            optimizer.step()

            #counting the losses
            training_loss += loss.item()


    #     forward propagation on valid datasets by certain steps
            if steps % print_every == 0: 
                valid_loss = 0
                valid_accuracy = 0
                model = model.eval()

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs,labels = inputs.to(device),labels.to(device)
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps,labels)

                        valid_loss += batch_loss.item()
                        # calculate the accuracy of validation datasets
                        ps = torch.exp(log_ps)                
                        _,top_class = ps.topk(1,dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                val_accuracies.append( valid_accuracy / len(valid_loader))
                #print out loss and accuracy on training and validation datasets
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {training_loss/print_every:.3f}.. "
                      f"valid loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"valid accuracy: {valid_accuracy/len(valid_loader):.3f}")                
                ## save model.state.dict(giood enough)
                if (valid_accuracy/len(valid_loader))  > best_accuracy:
                    best_accuracy = valid_accuracy / len(valid_loader)
                    best_model_wts = copy.deepcopy(model.state_dict())
                #go back go train mode
                model.train()
                training_loss = 0

    print("Model training finished")
    print('traing time is', (time.time() -start) // 60,'min')
        # Output a graph of loss metrics overiods.
#      plt.yla"Accuracy")
#      plabel("Epochs")
#      plt.title("Los vs. print_steps")
#      plt.plot(val_accues, label="validation")
#      plt.legend()
    return model,val_accuracies

    

def save_check_point(model,train_dataset,check_point):
    
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    model.class_to_idx = train_dataset.class_to_idx
    model.cpu()
    torch.save({'arch': 'resnet101',
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx, 
            'optimizer_state_dict': optimizer.state_dict()},
            
            check_point)
    print('checkepoing file saved as',check_point)
    return check_point 

