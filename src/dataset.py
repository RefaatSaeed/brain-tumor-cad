import cv2, torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class BrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(self.df["class_label"].unique())
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row["file_path"]), cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.c2i[row["class_label"]], dtype=torch.long)

def get_loaders(df, batch_size=32):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tfm = T.Compose([T.ToPILImage(), T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    eval_tfm = T.Compose([T.ToPILImage(), T.Resize((224,224)), T.ToTensor(), T.Normalize(mean, std)])
    
    train_ds = BrainDataset(df[df["dataset_split"]=="train"], train_tfm)
    val_ds = BrainDataset(df[df["dataset_split"]=="val"], eval_tfm)
    test_ds = BrainDataset(df[df["dataset_split"]=="test"], eval_tfm)
    
    # num_workers=0 ensures stability in Colab
    return (DataLoader(train_ds, batch_size, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size, shuffle=False, num_workers=0),
            DataLoader(test_ds, batch_size, shuffle=False, num_workers=0),
            train_ds.classes)
