from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import  torch
import matplotlib.pyplot as plt  
# transfrom each image into tensor and normalized with mean and std 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1370), (0.3081))])
Batch_size = 32

train_data = torch.utils.data.DataLoader(datasets.MNIST('../data', train = True, download  = True, transform = transform ), 
                                         batch_size = Batch_size, 
                                         shuffle = True
                                         )

test_data = torch.utils.data.DataLoader(datasets.MNIST('../data', train = False, download  = True, transform = transform), 
                                        batch_size = Batch_size, 
                                        shuffle = True)

# initailize the weights randomly 
weights = torch.randn(784, 10, requires_grad = True)
print(weights.shape)
def test(weights, test_data):
        test_size = len(test_data.dataset)
        correct = 0
        for batch_idx, (data, target) in enumerate(test_data):
            # print(batch_idx, data.shape, target.shape)
            data = data.view((-1, 28*28)) # reshape data in pytorch done by the function view 
            # print(data.shape)
            # break
            output = torch.matmul(data, weights)
            #print(output.shape, output[0])
            softmax = F.softmax(output, dim = 1)
            #print(softmax[0])
            pred = softmax.argmax(dim = 1, keepdims = True)
            # print(pred.eq(target.view_as(pred)))
            # break
            n_correct = pred.eq(target.view_as(pred)).sum().item() # test the difference between the target and the predections 
            correct += n_correct
        acc = correct / test_size
        print(' \r acc on test set', acc)
        return acc 
            
#test(weights, test_data)

# training NN in pytorch 

it = 0

for batch_idx, (data, target) in enumerate(train_data): 
    
    if weights.grad is not None: 
        weights.grad.zero_()
    
    # reshape data 
    data = data.view(-1, 28*28)
    #print(data.shape)
    output = torch.matmul(data, weights)
    #print("output shape :  {}".format(output.shape))
    log_softmax = F.log_softmax(output, dim = 1)
    # print('log softmax.shape {}'.format(log_softmax.shape))
    # break 
    loss = F.nll_loss(log_softmax, target) # calculate the negative log likelhod 
    print("\r loss shape  {}".format(loss), end = "")
    # compute the gradient for each varaibles 
    loss.backward()
    with torch.no_grad(): 
        weights -= 0.1*weights.grad
        
    it += 1 
    if it % 100 == 0:
        test(weights, test_data)
        
    if it > 100: 
        break  
        
batch_idx, (data, target) = next(enumerate(test_data))
data = data.view(-1, 28*28)
output = torch.matmul(data, weights)
softmax = F.softmax(output, dim = 1)
pred = softmax.argmax(dim = 1, keepdim = True)

plt.imshow(data[0].view(28, 28), cmap = "gray")
plt.title("prediction class {}".format(pred[0]))
plt.show()