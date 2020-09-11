import os
import random
import torch
import numpy as np
from utils.metrics import ConfusionMatrixCls


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, path, scores_node, scores_edge, alpha, task_name, epoch, best_pred):
    if scores_node['macro_f1'] + alpha*scores_edge['macro_f1'] > best_pred:
        best_pred = scores_node['macro_f1'] + alpha*scores_edge['macro_f1']
        save_path = os.path.join(path, "%s-%d-%.4f-%.4f.pth"%(task_name, epoch, scores_node['macro_f1'], scores_edge['macro_f1']))
        torch.save(model.state_dict(), save_path)
    
    return best_pred


def write_log(f_log, train_scores_node, train_scores_edge, val_scores_node, val_scores_edge, epoch, num_epochs):
    log = ""
    log = log + 'epoch [{}/{}] F1 Node: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, train_scores_node['macro_f1'], val_scores_node['macro_f1']) + "\n"
    log = log + 'epoch [{}/{}] F1 Edge: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, train_scores_edge['macro_f1'], val_scores_edge['macro_f1']) + "\n"
    log = log + "[train node] Precision = " + str(train_scores_node['precision']) + "\n"
    log = log + "[train node] Recall = " + str(train_scores_node['recall']) + "\n"
    log = log + "[train node] f1 = " + str(train_scores_node['f1']) + "\n"
    log = log + "[train edge] Precision = " + str(train_scores_edge['precision']) + "\n"
    log = log + "[train edge] Recall = " + str(train_scores_edge['recall']) + "\n"
    log = log + "[train edge] f1 = " + str(train_scores_edge['f1']) + "\n"
    log = log + "------------------------------------ \n"
    log = log + "[val node] Precision = " + str(val_scores_node['precision']) + "\n"
    log = log + "[val node] Recall = " + str(val_scores_node['recall']) + "\n"
    log = log + "[val node] f1 = " + str(val_scores_node['f1']) + "\n"
    log = log + "[val edge] Precision = " + str(val_scores_edge['precision']) + "\n"
    log = log + "[val edge] Recall = " + str(val_scores_edge['recall']) + "\n"
    log = log + "[val edge] f1 = " + str(val_scores_edge['f1']) + "\n"
    log += "================================\n"
    print(log)

    f_log.write(log)
    f_log.flush()


def write_summaryWriter(writer, loss, optimizer, train_scores_node, train_scores_edge, val_scores_node, val_scores_edge, epoch):
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalars('F1 Node', {'train': train_scores_node['macro_f1'], 'val': val_scores_node['macro_f1']}, epoch)
    writer.add_scalars('F1 Edge', {'train': train_scores_edge['macro_f1'], 'val': val_scores_edge['macro_f1']}, epoch)
    writer.add_scalars('Precision Node', {'train': train_scores_node['macro_precision'], 'val': val_scores_node['macro_precision']}, epoch)
    writer.add_scalars('Precision Edge', {'train': train_scores_edge['macro_precision'], 'val': val_scores_edge['macro_precision']}, epoch)
    writer.add_scalars('Recall Node', {'train': train_scores_node['macro_recall'], 'val': val_scores_node['macro_recall']}, epoch)
    writer.add_scalars('Recall Edge', {'train': train_scores_edge['macro_recall'], 'val': val_scores_edge['macro_recall']}, epoch)
            

class Trainer():
    def __init__(self, criterion1, criterion2, optimizer, n_class, device, alpha=1.0):
        self.criterion_node = criterion1
        self.criterion_edge = criterion2
        self.optimizer = optimizer
        self.n_class = n_class
        self.device = device
        self.alpha = alpha
        self.metrics_node = ConfusionMatrixCls(n_class)
        self.metrics_edge = ConfusionMatrixCls(2)
    
    def get_scores(self):
        score_node = self.metrics_node.get_scores()
        socre_edge = self.metrics_edge.get_scores()

        return score_node, socre_edge
    
    def reset_metrics(self):
        self.metrics_node.reset()
        self.metrics_edge.reset()
    
    def train(self, sample, model):
        model.train()
        sample = sample.to(self.device)
        label_node = sample.y.to(self.device)
        label_edge = sample.edge_label.to(self.device)

        node, edge = model(sample)
        loss_node = self.criterion_node(node, label_node)
        loss_edge = self.criterion_edge(edge, label_edge)
        loss = loss_node + self.alpha * loss_edge
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pred_node = np.argmax(node.cpu().detach().numpy(), axis=1)
        pred_edge = edge.cpu().detach().numpy()
        pred_edge = np.array(pred_edge>0.5, dtype='uint8')
        self.metrics_node.update(label_node.cpu().numpy(), pred_node)
        self.metrics_edge.update(label_edge.cpu().numpy(), pred_edge)

        return loss, loss_node, loss_edge


class Evaluator():
    def __init__(self, n_class, device):
        self.n_class = n_class
        self.device = device
        self.metrics_node = ConfusionMatrixCls(n_class)
        self.metrics_edge = ConfusionMatrixCls(2)
    
    def get_scores(self):
        score_node = self.metrics_node.get_scores()
        socre_edge = self.metrics_edge.get_scores()

        return score_node, socre_edge
    
    def reset_metrics(self):
        self.metrics_node.reset()
        self.metrics_edge.reset()
    
    def eval(self, sample, model):
        sample = sample.to(self.device)
        label_node = sample.y.to(self.device)
        label_edge = sample.edge_label.to(self.device)

        with torch.no_grad():
            node, edge = model(sample)
            pred_node = np.argmax(node.cpu().detach().numpy(), axis=1)
            pred_edge = edge.cpu().detach().numpy()
            pred_edge = np.array(pred_edge>0.5, dtype='uint8')
            self.metrics_node.update(label_node.cpu().numpy(), pred_node)
            self.metrics_edge.update(label_edge.cpu().numpy(), pred_edge)

        return pred_node, pred_edge
    
    def test(self, sample, model):
        sample = sample.to(self.device)
        with torch.no_grad():
            node, edge = model(sample)
            pred_node = np.argmax(node.cpu().detach().numpy(), axis=1)
            pred_edge = edge.cpu().detach().numpy()
            pred_edge = np.array(pred_edge>0.5, dtype='uint8')
        
        return pred_node, pred_edge
