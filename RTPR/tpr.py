import torch
from torch import nn
import pandas as pd
import random
from torch.optim.lr_scheduler import OneCycleLR

left_role = 0
right_role = 1

df = pd.read_csv("reversal_data.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
print(len(df))

#String to list
df["source"] = df["source"].apply(lambda x: eval(x))
df["context_vector"] = df["context_vector"].apply(lambda x: eval(x))

df_sorted = df.sort_values(by="source", key=lambda col: col.map(len))
#Allow testing of longer sequences
df_sorted.drop(df_sorted.tail(1000).index, inplace=True)

model_size = len(df_sorted["context_vector"].iloc[0])

def right_recurse(lst):
    batch_size = lst[0].shape[0]
    tensors = [[x, torch.full((batch_size,), left_role)] for x in lst]
    
    acc = [tensors[-1]]
    for t in reversed(tensors[:-1]):
        acc = [t, [acc, torch.full((batch_size,), right_role)]]
    return acc

def left_recurse( lst):
    batch_size = lst[0].shape[0]
    tensors = [[x, torch.full((batch_size,), right_role)] for x in lst]
    
    acc = [tensors[0]]
    for t in tensors[1:]:
        acc = [[acc, torch.full((batch_size,), left_role)], t]
    return acc

def collate_lists(*lists):
    assert len(set(len(l) for l in lists)) == 1, "All lists must be the same length"
    
    return [torch.tensor([lst[i] for lst in lists]) for i in range(len(lists[0]))]

#Evaluate a model (for validation and testing loss)
def evaluate(model, src, trg, criterion, device):
    with torch.no_grad():
        model.eval()
        epoch_loss = 0.0

        for i in range(len(src)):
            src_batch = src[i]
            trg_batch = trg[i]
            # src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            encoding = model(src_batch)
            output = torch.matmul(encoding, model.wo.T)    # (batch_size, model_size)
            loss = criterion(output, trg_batch)
            epoch_loss += loss.item()

        return epoch_loss / len(src)

batch_size = 5

sources = df_sorted["source"].tolist()
targets = df_sorted["context_vector"].tolist()

src_batches = []
trg_batches = []

for i in range(0, len(df_sorted), batch_size):
    src_batch = sources[i:i + batch_size]
    trg_batch = targets[i:i + batch_size] 

    collated = collate_lists(*src_batch)

    #Left or right branching
    nested   = right_recurse(collated)   
    # nested = left_recurse(collated)

    src_batches.append(nested)
    trg_batches.append(torch.stack([torch.tensor(v, dtype=torch.float32) for v in trg_batch]))

# 80/10/10 split
n       = len(src_batches)
n_train = int(0.8 * n) 
n_val   = int(0.1 * n) 

combined = list(zip(src_batches, trg_batches))
random.shuffle(combined)
src_batches, trg_batches = zip(*combined)

train_src, train_trg = src_batches[:n_train], trg_batches[:n_train]
val_src, val_trg = src_batches[n_train:n_train+n_val], trg_batches[n_train:n_train+n_val]
test_src, test_trg = src_batches[n_train+n_val:], trg_batches[n_train+n_val:]

class RTPREncode(nn.Module):
    def __init__(self, nfillers, nroles, filler_size, role_size, batch_size, model_size):
        super(RTPREncode, self).__init__()
        self.fillers = nn.Embedding(nfillers, filler_size)
        self.roles = nn.Embedding(nroles, role_size)
        self.filler_size = filler_size
        self.role_size = role_size
        self.model_size = model_size
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.4)
        self.w = torch.nn.Parameter(torch.randn(filler_size, filler_size*role_size)*0.01)
        self.wo = torch.nn.Parameter(torch.randn(model_size, filler_size)*0.01)

    def forward(self, x):
        result = torch.zeros(self.batch_size, self.filler_size, self.role_size, device=self.w.device)
    
        for pair in x:
            filler = pair[0]
            role = pair[1]
            
            if isinstance(filler, list):
                filler = self(filler)
            else:
                filler = self.fillers(filler)
            
            role = self.roles(role)
            result += torch.einsum("bf,br->bfr", filler, role)
        
        result = result.view(self.batch_size, -1)
        result = torch.einsum("ij,bj->bi", self.w, result) 
        return result

num_epochs = 75
model = RTPREncode(61, 2, 128, 64, batch_size, model_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = OneCycleLR(optimizer, max_lr=5e-3, steps_per_epoch=len(train_src), epochs=num_epochs)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} / {num_epochs}")
    model.train()

    epoch_loss = 0.0
    with torch.set_grad_enabled(True):
      for i in range(len(train_src)):
          src_batch = train_src[i]
          trg_batch = train_trg[i]
          optimizer.zero_grad()

          encoding = model(src_batch)
          output = torch.matmul(encoding, model.wo.T)

          loss = criterion(output, trg_batch)
          loss.backward()

          #Norm=magnitude, clipping gradient to ensure they don't explode
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
          optimizer.step()
          epoch_loss += loss.item()
          scheduler.step()

    print(f"Train Loss: {epoch_loss/len(train_src)}")
    val_loss = evaluate(model, val_src, val_trg, criterion, device)
    print(f"Validation Loss: {val_loss}")

torch.save(model.state_dict(), "RBT_rev.pt")