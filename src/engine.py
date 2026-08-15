import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

def train_epoch(model, loader, crit, opt, scaler, dev):
    model.train(); run_loss = 0.0; preds, targs = [], []
    for imgs, lbls in tqdm(loader, desc="Train", leave=False):
        imgs, lbls = imgs.to(dev), lbls.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            outs = model(imgs); loss = crit(outs, lbls)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        run_loss += loss.item() * imgs.size(0)
        preds.extend(torch.argmax(outs, 1).cpu().numpy()); targs.extend(lbls.cpu().numpy())
    return run_loss/len(loader.dataset), f1_score(targs, preds, average="macro")*100

@torch.no_grad()
def eval_epoch(model, loader, crit, dev):
    model.eval(); run_loss = 0.0; preds, targs = [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(dev), lbls.to(dev)
        outs = model(imgs); loss = crit(outs, lbls)
        run_loss += loss.item() * imgs.size(0)
        preds.extend(torch.argmax(outs, 1).cpu().numpy()); targs.extend(lbls.cpu().numpy())
    return run_loss/len(loader.dataset), f1_score(targs, preds, average="macro")*100
