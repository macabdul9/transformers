from config import args
# data
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
   
    
    
    
def data_loaders(device, exts = None, batch_first=True):
    """
    This function will return DataLoaders 

    Args:
    device (device type): tensor will be on cpu or gpu

    Returns: Multiple dataloader generator objects
    """

    # create data fields for source and target
    source = Field(
        init_token="<sos>",
        eos_token="<eos>",
        lower=True,
        tokenize="spacy",
        tokenizer_language="de",
        batch_first=batch_first
    )

    target = Field(
        init_token="<sos>",
        eos_token="<eos>",
        lower=True,
        tokenize="spacy",
        tokenizer_language="de",
        batch_first=batch_first
    )
    
    # download the dataset
    train, val, test = Multi30k.splits(
        exts=(".de", ".en"),
        fields=(source, target)
    )
    
    # build the vocab
    source.build_vocab(train)
    target.build_vocab(train)
    
    
    train_loader, val_loader, test_loader = BucketIterator.splits(
        datasets=(train, val, test),
        batch_size=args['batch_size'],
        device=device,
        shuffle=True
    )
    
    return source, target, train_loader, val_loader, test_loader
        