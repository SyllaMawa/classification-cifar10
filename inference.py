from dataset import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classif()
model.load_state_dict(torch.load("C:/Users/lekyo/OneDrive/Documents/ESIR3/Training/Projets/From_Scratch/models/model50.pt", weights_only=True))

model.to(device)
model.eval()
comp_corrects, total = 0, 0

for idx, (images, labels) in enumerate(testloader):
    print(f"Batch index -----------------------------> {idx}")
    #print(labels)

    outputs = model(images) #batch_size 32 32
    _, preds = torch.max(outputs, dim=1)

    #print(f"Shape of outputs : {outputs.shape}")
    #print(f"Shape of labels : {labels.shape}")
    #print(labels.dtype)
    #print(labels[0])

    comp_corrects += (preds == labels).sum().item()
    total += labels.size(0)

    #print(f"Shape of comp : {comp.shape}")

print(f"Comp_corrects : {comp_corrects}")

accuracy = (comp_corrects / total) * 100

print(f"Accuracy of the classification model : {accuracy}%")



