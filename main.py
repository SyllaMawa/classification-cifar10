from dataset import *
from model import Classif
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#optimizer
criterion = nn.CrossEntropyLoss()

#Model
model = Classif()
model.to(device)

#Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

writer = SummaryWriter("logs/it6")

#loop
num_epoch = 101

for epoch in tqdm(range(num_epoch)):
    model.train()
    tot_loss = 0
    tot_lossv = 0
    for idx, (img, label) in enumerate(trainloader):

        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(img) #out : tens 4 10
        #print(f"Out: {out.shape}")

        #_, label_out = torch.max(out, dim=1) #indice du maximum des logits

        #print(f"Label: {label_out}")
        #loss = criterion(classes.index(label), label_out) # on recupere l'index du label
        #vérif
        print(f"Out shape: {out.shape}, Label shape: {label.shape}")
        print(f"Out mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")

        #
        loss = criterion(out, label) # on recupere l'index du label

        loss.backward()
        optimizer.step()

        tot_loss += loss.item()

        if idx == 0:
            writer.add_images("Images/train_batch", img, epoch)
    
    writer.add_scalar("Loss/train", tot_loss / len(trainloader), global_step=epoch)
    

    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            for idx_val, (img_val, label_val) in enumerate(testloader):
                img_val, label_val = img_val.to(device), label_val.to(device)
                out_val = model(img_val)

                #_, label_outv = torch.max(out_val, dim=1)

                loss_val = criterion(out_val, label_val)

                tot_lossv += loss_val.item()

            writer.add_scalar("Loss/val", tot_lossv / len(testloader), global_step=epoch // 2)
            model.train()

    writer.flush()

    if epoch % 50 == 0:
        torch.save(model.state_dict(),f"C:/Users/lekyo/OneDrive/Documents/ESIR3/Training/Projets/From_Scratch/models/model{epoch}.pt")

writer.close()
            







