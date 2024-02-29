import time
import torch
from tqdm import tqdm
from utils.helpers import list_of_distances, make_one_hot

from utils.log import create_logger

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True, cfg=None,
                   coefs=None, estimator=None, log=print, epoch=0, t_epoch=100, tcurent_epoch=0):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_l1_norm = 0
    total_proto_loss = 0

    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_regs, reg_ = 0, 0
    proto_regs = 0

    epsilon = 1e-4

    reg = cfg.train.reg
    reg_epoch = cfg.train.reg_epoch

    if is_train:
        progress = tqdm(enumerate(dataloader))
    else:
        progress = enumerate(dataloader)

    for i, (image, label) in progress:
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances, l1_norm = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)             

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                # prototypre dissimilarity
                proto_vect = model.module.prototype_vectors
                if cfg.train.reg_proto:
                    proto_similarity = 0
                    for ii in range(len(proto_vect)):
                        for jj in range(len(proto_vect)):
                            proto_similarity += torch.sqrt(torch.sum((proto_vect[ii] - proto_vect[jj] + epsilon) ** 2))
                    ## since the similarity matrix is symetric
                    proto_regs = cfg.train.reg_proto * torch.log(1.0001 + (proto_similarity/2)) 
                else:
                    proto_regs = torch.tensor([0], device='cuda')               

                if cfg.train.fc_classification_layer:
                    class_weight = model.module.last_layer.weight
                else:
                    weight = model.module.last_layer
                    weight = tuple([weight[:, kk] for kk in range(cfg.data.num_classes)])
                    concat_weight = torch.cat(weight).unsqueeze(0)
                    class_weight = concat_weight.repeat(cfg.data.num_classes, 1)
                
                if cfg.train.reg_fcl:
                    if use_l1_mask:
                        l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                        l1 = (class_weight * l1_mask).norm(p=1)
                    else:
                        l1 = class_weight.norm(p=1)
                else:
                    l1 = 0

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1) if cfg.train.fc_classification_layer else 0

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_proto_loss += proto_regs.item()
            total_cluster_cost += cluster_cost.item()
            total_l1_norm += l1_norm.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

            estimator.update(output, label)

            # compute gradient and do SGD step            
            if reg_epoch: # regularize from a given epoch           
                if epoch >= reg_epoch:
                    regs = reg * torch.log(l1_norm)
                    reg_ = torch.tensor([1])
                else:
                    regs = 0 * torch.log(l1_norm)
                    reg_ = torch.log(l1_norm)
            else: # regularrize at every epoch
                regs = reg * torch.log(l1_norm)
                reg_ = torch.log(l1_norm)
            total_regs += regs.item()
            reg_ = reg_.item()               
            
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1
                          + regs 
                          + proto_regs )  # reg * l1_norm
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 + regs + proto_regs # reg * l1_norm 
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1
                          + regs
                          + proto_regs) #reg * l1_norm
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1 + regs + proto_regs #reg * l1_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del output
        del predicted
        del min_distances

        avg_regs = total_regs / n_batches
        avg_CE = total_cross_entropy / n_batches
        avg_proto_sim = total_proto_loss / n_batches
        avg_l1_norm = total_l1_norm / n_batches
        avg_accuracy = n_correct / n_examples * 100
        avg_cluster_cost = total_cluster_cost / n_batches
        avg_separation_cost = total_separation_cost / n_batches
        avg_total_separation_cost = total_avg_separation_cost / n_batches
        
        if is_train: 
            if cfg.data.binary:
                progress.set_description('epoch: [{} / {}], bash: {}/{}, CE_loss: {:.5f}, acc: {:.4f}, AUC: {:.4f}, clst_cost: {:.4f}, T. sep_cost: {:.4f}, l1_norm: ({:.4f}, {:.4f}), proto_sim: {:.4f}'.format(
                    epoch, t_epoch, i, len(dataloader), avg_CE, avg_accuracy, estimator.get_auc_auprc(6)[0][0], avg_cluster_cost, avg_total_separation_cost, avg_regs, reg_, avg_proto_sim)) 
            else:
                progress.set_description('epoch: [{} / {}], bash: {}/{}, CE_loss: {:.5f}, acc: {:.4f}, kappa: {:.4f}, bin_acc.: {:.4f}, AUC: {:.4f}, clst_cost: {:.4f}, T. sep_cost: {:.4f}, l1_norm: {:.4f}, proto_sim: {:.4f}'.format(
                    epoch, t_epoch, i, len(dataloader), avg_CE, avg_accuracy, estimator.get_kappa(6), estimator.get_accuracy(6)[1], estimator.get_auc_auprc(6)[0][0],  avg_cluster_cost, avg_total_separation_cost, avg_regs, avg_proto_sim))
                
        else:
            pass
    
    end = time.time(); t_CE = total_cross_entropy / n_batches
    t_proto_sim = total_proto_loss / n_batches
    t_cl_cost = total_cluster_cost / n_batches; acc = n_correct / n_examples * 100

    log('\tcross ent: \t{0}'.format(t_CE))
    log('\tproto sim: \t{0}'.format(t_proto_sim))
    log('\tcluster: \t{0}'.format(t_cl_cost)); 

    if class_specific:
        t_sep_loss = total_separation_cost / n_batches; avg_sep = total_avg_separation_cost / n_batches
        #log('\tseparation:\t{0}'.format(t_sep_loss))   
        log('\tavg separation:\t{0}'.format(avg_sep))
    
    log('\taccu: \t\t{0}%'.format(acc))
    #log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()

    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))

    log('\tavg reg: \t\t{0}%'.format(avg_regs))
    
    return (acc/100, t_CE, l1_norm.cpu().item())

def train(model, dataloader, optimizer, class_specific=False, coefs=None, estimator=None, log=print, 
          epoch=None, cfg=None, t_epoch=None):
    
    assert(optimizer is not None)
    T_epoch = t_epoch if t_epoch else cfg.train.epochs # t_epoch is the total epochs for the FCL step 
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer, cfg=cfg, coefs=coefs,
                          class_specific=class_specific, estimator=estimator, log=log, epoch=epoch, t_epoch=T_epoch)


def test(model, dataloader, class_specific=False, log=print, estimator=None, cfg=None, tcurent_epoch=0):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None, estimator=estimator,
                          class_specific=class_specific, log=log, cfg=cfg, tcurent_epoch=tcurent_epoch)


def last_only(model, cfg, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False

    if cfg.train.fc_classification_layer:
        for p in model.module.last_layer.parameters(): # prototype learning 
            p.requires_grad = True  
    else:
        model.module.last_layer.requires_grad = True
    
    log('\tlast layer')

def warm_only(model, cfg, log=print): 
    '''
        only train the two additional conv layers and the protoype layer
    '''
    for p in model.module.features.parameters(): # backbone feature param @dj
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters(): # 2 1x1-conv layer parms @dj
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    if cfg.train.fc_classification_layer:
        for p in model.module.last_layer.parameters(): # prototype learning 
            p.requires_grad = True  
    else:
        model.module.last_layer.requires_grad = True  
    log('\twarm')


def joint(model, cfg, log=print):
    '''
        train all the conv layers and the protoype layer jointly
    '''
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True

    if cfg.train.fc_classification_layer:
        for p in model.module.last_layer.parameters(): # prototype learning 
            p.requires_grad = True  
    else:
        model.module.last_layer.requires_grad = True
    
    log('\tjoint')
