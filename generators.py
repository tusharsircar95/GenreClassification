import network, losses
import torch
import numpy as np
import torch.nn as nn
import visualizer

def generate_embeddings(train_dataset,val_dataset,device,embed_type,n_epochs=10,batch_size=32,save_path=None):
    
  train_dl = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
  val_dl = torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=batch_size)

  gnet = network.GenreNet(embed_type).to(device)
  
  if embed_type not in ['softmax','center-softmax','triplet','sphere','cos']:
    raise Exception('Invalid embedding type!')

  if embed_type == 'softmax':
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{'params': gnet.net.parameters(), 'lr': 1e-6},
                                {'params': gnet.classifier.parameters(), 'lr': 1e-3},
                                {'params': gnet.embedding_layer.parameters(), 'lr': 1e-3}])
  
  elif embed_type == 'center-softmax':
    criterion = torch.nn.CrossEntropyLoss()
    center_loss = losses.CenterLoss(num_classes=10, feat_dim=32, use_gpu=True)

    optimizer = torch.optim.Adam([{'params': gnet.net.parameters(), 'lr': 1e-5},
                                  {'params': gnet.classifier.parameters(), 'lr': 1e-3},
                                  {'params': gnet.embedding_layer.parameters(), 'lr': 1e-3},
                                  {'params': center_loss.parameters(), 'lr': 1e-3}])

  elif embed_type == 'cos':
    criterion = losses.CosLoss()
    optimizer = torch.optim.Adam([{'params': gnet.net.parameters(), 'lr': 1e-6},
                                {'params': gnet.classifier.parameters(), 'lr': 1e-3},
                                {'params': gnet.embedding_layer.parameters(), 'lr': 1e-3}])


  train_tracker = []
  val_tracker = []
  for epoch in range(n_epochs):    
    gnet.train()
    epoch_losses = []
    total_samples = 1e-5
    correct_samples = 0
    for i,batch in enumerate(train_dl):
      gnet.zero_grad()
      optimizer.zero_grad()
        
      X, y = batch[0].to(device), batch[1].to(device).long()
      embeddings, predictions = gnet(X)
      
      if embed_type == 'softmax':
        _, y_pred = torch.max(predictions,1)        
        total_samples += y.size(0)
        correct_samples += (y_pred == y).sum().item()
        
        loss = criterion(predictions,y.long())
      
      elif embed_type == 'sphere':
        loss, acc_batch = criterion(predictions,y.long())
        correct_samples += y.size(0) * acc_batch
        total_samples += y.size(0)
      
      elif embed_type == 'cos':
        loss, acc_batch = criterion(predictions,y.long())
        correct_samples += y.size(0) * acc_batch
        total_samples += y.size(0)
        
      elif embed_type == 'center-softmax':
        _, y_pred = torch.max(predictions,1)        
        total_samples += y.size(0)
        correct_samples += (y_pred == y).sum().item()

        closs = center_loss(embeddings, y) 
        loss = criterion(predictions,y.long()) + closs

      epoch_losses.append(loss.item())
      loss.backward()
      optimizer.step()

    epoch_loss = np.mean(epoch_losses)
    epoch_acc = correct_samples/total_samples
    train_tracker.append((epoch_loss,epoch_acc))
    if (epoch+1) % 5 == 0:
      print("Train Loss after epoch {} = {}".format(epoch,np.mean(epoch_losses)))
      print("Train Accuracy after epoch {} = {}".format(epoch,correct_samples/total_samples))
      #torch.save(gnet, './checkpoints/gnet_model_{}_epoch_{}.pth'.format(embed_type,epoch))

    epoch_losses = []
    total_samples = 1e-5
    correct_samples = 0
    gnet.eval()
    for i,batch in enumerate(val_dl):

      gnet.zero_grad()
      optimizer.zero_grad()

      X, y = batch[0].to(device), batch[1].to(device).long()
      embeddings, predictions = gnet(X)

      if embed_type == 'softmax':
        _, y_pred = torch.max(predictions,1)        
        total_samples += y.size(0)
        correct_samples += (y_pred == y).sum().item()
        
        loss = criterion(predictions,y.long())
            
      elif embed_type == 'sphere':
        loss, acc_batch = criterion(predictions,y.long())
        correct_samples += y.size(0) * acc_batch
        total_samples += y.size(0)
      
      elif embed_type == 'cos':
        loss, acc_batch = criterion(predictions,y.long())
        correct_samples += y.size(0) * acc_batch
        total_samples += y.size(0)

      elif embed_type == 'center-softmax':
        _, y_pred = torch.max(predictions,1)        
        total_samples += y.size(0)
        correct_samples += (y_pred == y).sum().item()

        closs = center_loss(embeddings, y) 
        loss = criterion(predictions,y.long()) + closs

      epoch_losses.append(loss.item())

    epoch_loss = np.mean(epoch_losses)
    epoch_acc = correct_samples/total_samples
    val_tracker.append((epoch_loss,epoch_acc))
    if (epoch + 1) % 5 == 0:
      print("Val Loss after epoch {} = {}".format(epoch,np.mean(epoch_losses)))
      print("Val Accuracy after epoch {} = {}".format(epoch,correct_samples/total_samples))
      print('\n')
      visualizer.visualize_embeddings(val_dl,gnet,device)

  if not save_path is None:
    
    train_embeddings = None
    for i,batch in enumerate(train_dl):
      X, y = batch[0].to(device), batch[1].to(device)
      embeddings, predictions = gnet(X)
      embeddings = embeddings.detach().cpu().numpy()
      if train_embeddings is None:
        train_embeddings = embeddings
      else: train_embeddings = np.concatenate([train_embeddings,embeddings])

    val_embeddings = None
    for i,batch in enumerate(val_dl):
      X, y = batch[0].to(device), batch[1].to(device)
      embeddings, predictions = gnet(X)
      embeddings = embeddings.detach().cpu().numpy()
      if val_embeddings is None:
        val_embeddings = embeddings
      else: val_embeddings = np.concatenate([val_embeddings,embeddings])

    np.save('{}/{}_{}.npy'.format(save_path,embed_type,'train'),train_embeddings)
    np.save('{}/{}_{}.npy'.format(save_path,embed_type,'val'),val_embeddings)
