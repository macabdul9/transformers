import torch
from tqdm import tqdm

def eval(model, data_loader, criterion):
    """
    Function for evaluation step

    Args:
        model ([]): tranformer model
        data_loader (BucketIterator): data_loader to evaluate
        criterion (Loss Object): criterion to calculate the loss 
    """
    
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            src, trg = batch.src, batch.trg
            batch_size, trg_len = trg.shape[0], trg.shape[1]
            outputs = model(src, trg)
            l = criterion(outputs.view(batch_size*trg_len, -1), trg.type_as(outputs).view(-1))
            losses.append(l.item())
    loss = sum(losses)/len(losses)
    ppl = torch.exp(torch.tensor(loss)).item() 
    return loss, ppl   




def train(model, train_loader, val_loader, criterion, optimizer, epochs = 10):
    """
    
    Function to train the model

    Args:
        model (nn.Module): model 
        train_loader (B): [description]
        val_loader ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): [description]
        
        epochs (int, optional): [description]. Defaults to 10.
    """
    
    epoch_progress = tqdm(total=epochs, desc="Epoch", position=0)
    total_steps = len(train_loader)*epochs
    steps = 0
    
    for epoch in range(epochs):
        
        train_loss = []
        for batch in train_loader:
            src, trg = batch.src, batch.trg
            batch_size, trg_len = batch.trg.shape[0], batch.trg.shape[1]
            outputs = model(src, trg)
            
            # compute the loss and backpropagate 
            loss = criterion(outputs.view(batch_size*trg_len, -1), trg.type_as(outputs).view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
            if steps % (len(train_loader)//2) == 0:
                ppl = torch.exp(loss).item()
                print(f'Steps {steps}/{total_steps} | Train_loss {loss.item():.4f} | Train_ppl {ppl:.4f}')
            
            train_loss.append(loss.item())
            steps += 1
        
        avg_loss = sum(train_loss)/len(train_loss)
        avg_ppl = torch.exp(torch.tensor([avg_loss]))
        
        val_loss, val_ppl = eval(model,  val_loader, criterion)
        
        print(f'Epoch {epoch}/{epochs} | Train_loss {avg_loss:.4f} | Train_ppl {avg_ppl:.4f} | Val_loss {val_loss:.4f} | Val_ppl {val_ppl:.4f}')
        
        epoch_progress.update(1)
    
    
    
    
