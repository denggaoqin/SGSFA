import torch
from impl import models




def train(optimizer, model, dataloader, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        # coreness=40
        pred, pos, dim= model(*batch[:-1], id=0)  # pred=(64,6)
        flag = models.FLAG(dim, loss_fn, optimizer)
        forward = lambda perturb: model(*batch[:-1], perturb, id=0)
        loss, out = flag(model, forward, pos, batch[-1])
        total_loss.append(loss.detach().item())
    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    for batch in dataloader:
        pred, _, _ = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)
