import torch
import argparse
import dataset
import generators
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_root', type=str, default=None,
                    help='root directory for the dataset')
    parser.add_argument('--preloaded',type=int, required=True,
                    help='is data already loaded in directory?')
    parser.add_argument('--n_epochs', type=int, default=10,
                    help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for training')
    parser.add_argument('--embedding_type', type=str, required=True,
                    help='loss function to train with')
    parser.add_argument('--embedding_save_path', type=str, default=None,
                    help='path to store embeddings')
    args = parser.parse_args()
    
    print(args.data_root,args.preloaded,args.n_epochs)
    if not args.preloaded:
        if args.data_root is None:
            raise Exception('data_root passed is None')
        song_data = dataset.prepare_data(args.data_root)
        print('{} songs loaded...'.format(len(song_data)))
        train_split, val_split = dataset.train_test_split(song_data,split_ratio=0.90)
    else: 
        song_data = None
        train_split = val_split = None

    train_dataset = dataset.MusicDataset(train_split,mode='train',preloaded=args.preloaded == 1)
    val_dataset = dataset.MusicDataset(val_split,mode='val',preloaded=args.preloaded == 1)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Create directory to save embeddings if it doesn't exist
    if not os.path.exists(args.embedding_save_path):
        os.mkdir(args.embedding_save_path)
        
    generators.generate_embeddings(n_epochs=args.n_epochs,
                                   batch_size=args.batch_size,
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   device=device,
                                   embed_type=args.embedding_type,
                                   save_path=args.embedding_save_path)