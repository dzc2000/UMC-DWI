import argparse
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score
from model import UMC, ll_loss, cons_gradient, getVac
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score
from metrics import expected_calibration_error
from dataset import MultiViewDataset
from utils.utils import *
from utils.logger import create_logger
import os
from torch.utils.data import DataLoader, random_split

    

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--view1_path", type=str, default="./dataset/DWIb500")
    parser.add_argument("--view2_path", type=str, default="./dataset/DWIb1500")
    parser.add_argument("--view3_path", type=str, default="./dataset/DWIb2000")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="ReleasedVersion")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/UMC/result/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_classes", type=int, default=3)
    parser.add_argument("--annealing_epoch", type=int, default=50)


def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_forward(i_epoch, model, args, ll_loss, batch):
    view1, view2, view3, tgt = batch['v1'], batch['v2'], batch['v3'], batch['label']

    view1, view2, view3, tgt = view1.cuda(), view2.cuda(), view3.cuda(), tgt.cuda()
    view1_alpha, view2_alpha, view3_alpha, fusion_alpha = model(view1, view2, view3)

    loss = ll_loss(tgt, view1_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
           ll_loss(tgt, view2_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
           ll_loss(tgt, view3_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
           ll_loss(tgt, fusion_alpha, args.n_classes, i_epoch, args.annealing_epoch)
    loss += (cons_gradient(tgt, view1_alpha, view2_alpha, args.n_classes) + cons_gradient(tgt, view1_alpha, view3_alpha, args.n_classes) + \
        cons_gradient(tgt, view2_alpha, view3_alpha, args.n_classes)) * 0.1
    return loss, view1_alpha, view2_alpha, view3_alpha, fusion_alpha, tgt


def model_eval(i_epoch, data, model, args, criterion):
    model.eval()
    with torch.no_grad():
        losses, view1_preds, view2_preds, view3_preds, fusion_preds, tgts, fusion_probs, confidences, predicteds = [], [], [], [], [], [], [], [], []
        for batch in data:
            threshold = 1.0
            #if u_f.min().item() < threshold:
            loss, view1_alpha, view2_alpha, view3_alpha, fusion_alpha, tgt = model_forward(i_epoch, model, args, criterion, batch)
            fusion_uncertainty = getVac(fusion_alpha)
            if torch.any(fusion_uncertainty < threshold):
                index = (fusion_uncertainty < threshold).view(-1)
                view1_alpha, view2_alpha, view3_alpha, fusion_alpha = view1_alpha[index], view2_alpha[index], view3_alpha[index], fusion_alpha[index]
                tgt = tgt[index]
            losses.append(loss.item())
            fusion_prob = (fusion_alpha/torch.sum(fusion_alpha, dim=1, keepdim=True))
            confidence, predicted = torch.max(fusion_prob.data, 1)
            view1_pred = view1_alpha.argmax(dim=1).cpu().detach().numpy()
            view2_pred = view2_alpha.argmax(dim=1).cpu().detach().numpy()
            view3_pred = view3_alpha.argmax(dim=1).cpu().detach().numpy()
            fusion_pred = fusion_alpha.argmax(dim=1).cpu().detach().numpy()
            
            view1_preds.append(view1_pred)
            view2_preds.append(view2_pred)
            view3_preds.append(view3_pred)
            fusion_preds.append(fusion_pred)
            confidences.append(confidence)
            predicteds.append(predicted)
            fusion_probs.append(fusion_prob.cpu().detach().numpy())
            
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}

    tgts = [l for sl in tgts for l in sl]
    view1_preds = [l for sl in view1_preds for l in sl]
    view2_preds = [l for sl in view2_preds for l in sl]
    view3_preds = [l for sl in view3_preds for l in sl]
    fusion_probs = [l for sl in fusion_probs for l in sl]
    fusion_preds = [l for sl in fusion_preds for l in sl]
    confidences = [l for sl in confidences for l in sl]
    predicteds = [l for sl in predicteds for l in sl]
            
    metrics["f1"] = f1_score(tgts, fusion_preds, average='macro')
    metrics["r"] = recall_score(tgts, fusion_preds, average='macro')
    metrics["p"] = precision_score(tgts, fusion_preds, average='macro')
    metrics["auc"] = roc_auc_score(tgts, fusion_probs, multi_class='ovr')
    metrics["ece"] = expected_calibration_error(confidences, predicteds, tgts).item()
    metrics["view1_acc"] = accuracy_score(tgts, view1_preds)
    metrics["view2_acc"] = accuracy_score(tgts, view2_preds)
    metrics["view3_acc"] = accuracy_score(tgts, view3_preds)
    metrics["fusion_acc"] = accuracy_score(tgts, fusion_preds)
    
    
    return metrics


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    view1_mean = [0.2898, 0.2898, 0.2898]
    view1_std = [0.1993, 0.1993, 0.1993]
    
    view2_mean = [0.1631, 0.1631, 0.1631]
    view2_std = [0.1173, 0.1173, 0.1173]
    
    view3_mean = [0.1360, 0.1360, 0.1360]
    view3_std = [0.0929, 0.0929, 0.0929]
    
    view1_transform = list()
    view1_transform.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    view1_transform.append(transforms.ToTensor())
    view1_transform.append(transforms.Normalize(mean=view1_mean, std=view1_std))
    
    view2_transform = list()
    view2_transform.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    view2_transform.append(transforms.ToTensor())
    view2_transform.append(transforms.Normalize(mean=view2_mean, std=view2_std))
    
    view3_transform = list()
    view3_transform.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    view3_transform.append(transforms.ToTensor())
    view3_transform.append(transforms.Normalize(mean=view3_mean, std=view3_std))

    dataset = MultiViewDataset([args.view1_path, args.view2_path, args.view3_path], 
                               [transforms.Compose(view1_transform),
                               transforms.Compose(view2_transform),
                               transforms.Compose(view3_transform)])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = UMC(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, view1_alpha, view2_alpha, view3_alpha, fusion_alpha, tgt = model_forward(i_epoch, model, args, ll_loss, batch)
            if args.gradient_accumulation_steps > 1:
                 loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(
            np.inf, test_loader, model, args, ll_loss
        )
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("val", metrics, logger)
        logger.info(
        "{}: Loss: {:.5f} | view1_acc: {:.5f}, view2_acc: {:.5f}, view3_acc: {:.5f}, fusion acc: {:.5f}, f1: {:.5f}, r: {:.5f}, p: {:.5f}, auc: {:.5f}, ece: {:.5f}".format(
            "val", metrics["loss"], metrics["view1_acc"], metrics["view2_acc"], metrics["view3_acc"],
            metrics["fusion_acc"], metrics["f1"], metrics["r"], metrics["p"], metrics["auc"], metrics["ece"]
        )
        )
        tuning_metric = metrics["fusion_acc"]
        
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break
    
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = model_eval(
        np.inf, test_loader, model, args, ll_loss
    )
    logger.info(
        "{}: Loss: {:.5f} | view1_acc: {:.5f}, view2_acc: {:.5f}, view3_acc: {:.5f}, fusion acc: {:.5f}, f1: {:.5f}, r: {:.5f}, p: {:.5f}, auc: {:.5f}, ece: {:.5f}".format(
            "Test", test_metrics["loss"], test_metrics["view1_acc"], test_metrics["view2_acc"], test_metrics["view3_acc"],
            test_metrics["fusion_acc"], test_metrics["f1"], test_metrics["r"], test_metrics["p"], test_metrics["auc"], test_metrics["ece"]
        )
    )
    log_metrics(f"Test", test_metrics, logger)
    

def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
