import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



def validation(model, testloader, criterion, device='cpu'):
    accuracy = 0
    test_loss = 0
    num = 0
    for inputs, labels in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        batch_size = inputs.shape[0]
        seql = inputs.shape[1]
        seq_lens = torch.tensor([seql] * batch_size, dtype=torch.float)
        output = model.forward(inputs, seq_lens)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(model, trainloader, validloader, criterion, optimizer, 
          epochs=10, print_every=10, device='cpu', run_name='model_mlstm_fcn'):
    print("Training started on device: {}".format(device))

    valid_loss_min = np.Inf # track change in validation loss
    steps = 0
    
    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for inputs, labels in trainloader:
            steps += 1

            inputs = inputs.float()
            inputs, labels = inputs.to(device),labels.to(device)
            
            optimizer.zero_grad()
            batch_size = inputs.shape[0]
            seql = inputs.shape[1]
            seq_lens = torch.tensor([seql] * batch_size, dtype=torch.float)
            output = model.forward(inputs, seq_lens)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.6f}.. ".format(train_loss/print_every),
                      "Val Loss: {:.6f}.. ".format(valid_loss/len(validloader)),
                      "Val Accuracy: {:.2f}%".format(accuracy/len(validloader)*100))
                
                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                    torch.save(model.state_dict(), 'model/'+run_name+'.pt')
                    valid_loss_min = valid_loss

                train_loss = 0

                model.train()


def load_ISLD_datasets(dataset_name='ISLD'):
    data_path = './datasets/'+dataset_name+'/'

    X_train = torch.load(data_path+'X_train_tensor.pt')
    X_val = torch.load(data_path+'X_val_tensor.pt')
    X_test = torch.load(data_path+'X_test_tensor.pt')

    y_train = torch.load(data_path+'y_train_tensor.pt')
    y_val = torch.load(data_path+'y_val_tensor.pt')
    y_test = torch.load(data_path+'y_test_tensor.pt')

    seq_lens_train = torch.load(data_path+'seq_lens_train_tensor.pt')
    seq_lens_val = torch.load(data_path+'seq_lens_val_tensor.pt')
    seq_lens_test = torch.load(data_path+'seq_lens_test_tensor.pt')

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, seq_lens_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val, seq_lens_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test, seq_lens_test)

    return train_dataset, val_dataset, test_dataset

def load_npy_datasets(dataset_name='BipedalWalkerHC'):
    data_path = './datasets/'+dataset_name+'/'

    feature = np.load(data_path+'feature.npy')
    X_test = np.load(data_path+'X_test.npy')

    labels = np.load(data_path+'labels.npy')
    y_test = np.load(data_path+'y_test.npy')

    X_train, X_val, y_train, y_val = train_test_split(feature, labels, test_size=0.2)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    return train_dataset, val_dataset, test_dataset

def load_datasets(dataset_name='BipedalWalkerHC'):
    data_path = './datasets/'+dataset_name+'/'

    X_train = np.load(data_path+'X_train.npy')
    X_test = np.load(data_path+'X_test.npy')

    y_train = np.load(data_path+'y_train.npy')
    y_test = np.load(data_path+'y_test.npy')
    
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    return train_dataset, val_dataset, test_dataset
