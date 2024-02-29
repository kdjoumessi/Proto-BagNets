import os
import copy
import torch
import pandas as pd
import utils.save as save

import training.push as push
from training.init_train import *
import training.train_and_test as tnt
from utils.preprocess import preprocess_input_function


def train(cfg, model, multi_model, dataset, paths, estimator, logs, debug=False):
    log, logclose, logger = logs

    push_epochs = [] 
    train_fcl_epoch = 0   
    coefs=cfg.solver.coefs    
    model_dir = cfg.paths.model_save_path
    class_specific = cfg.train.class_specific

    list_push = ([], [])
    list_lnorm = ([], [])
    list_train, list_test = ([], []), ([], [])
    list_train_fcl, list_test_fcl = ([], [], [], [], []), ([], [], [], [], []) 
    push_nun_kappa_auc, max_FCL_kappa_auc, best_epoch_val_kappa_auc = -1, -1, -1
    max_accu, max_push_accu, max_FCL_accu, push_nun, best_epoch_val_acc = -1, -1, -1, -1, -1

    ############ initialization ############
    train_dataset, push_dataset, val_dataset, dset_test = dataset

    train_loader, train_push_loader, val_loader = initialize_dataloader(cfg, train_dataset, push_dataset, val_dataset)

    warm_optimizer, joint_optimizer, last_layer_optimizer, all_layer_optimizer = initialize_optimizer(cfg, model) 
    joint_lr_scheduler, all_lr_scheduler = initialize_lr_scheduler(cfg, joint_optimizer, all_layer_optimizer)

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    print(f'training set size: {len(dataset[0])}, \t training loader size: {len(train_loader)}')
    print(f'validation set size: {len(dataset[2])}, \t val loader size: {len(val_loader)}')

    # train the model
    log('start training')
    train_estimator, test_estimator = estimator

    train_fcl_epoch = cfg.train.train_last_layer_epoch
    push_start = cfg.train.push_start
    push_epochs = [i for i in range(cfg.train.epochs) if i % push_start ==0] 

    if debug:
        push_epochs.append(1)
 
    for epoch in range(1, cfg.train.epochs+1):
        log('epoch: \t{0}'.format(epoch))
        train_estimator.reset(); test_estimator.reset()

        # warmup scheduler update
        if (epoch < cfg.train.warm_epochs): # 
            # freeze the feature parameters (requires_grad=False, ImageNet) and train the others parameters
            print('warm')
            tnt.warm_only(model=multi_model, cfg=cfg, log=log) # only update the 1x1 conv layer and prototype params
            train_metrics = tnt.train(cfg=cfg, model=multi_model, dataloader=train_loader, optimizer=warm_optimizer, 
                        class_specific=class_specific, coefs=coefs, estimator=train_estimator, log=log, epoch=epoch)
        else:
            print('joint')
            tnt.joint(model=multi_model, cfg=cfg, log=log) # update all params except the FCL 
            train_metrics = tnt.train(cfg=cfg, model=multi_model, dataloader=train_loader, optimizer=joint_optimizer, 
                    class_specific=class_specific, coefs=coefs, estimator=train_estimator, log=log, epoch=epoch)
        
        test_metrics = tnt.test(model=multi_model, dataloader=val_loader, class_specific=class_specific, estimator=test_estimator, log=log, cfg=cfg)

        list_train[0].append(train_metrics[0]*100); list_train[1].append(train_metrics[1])
        list_test[0].append(test_metrics[0]*100); list_test[1].append(test_metrics[1]*100)
        
        if test_metrics[0] > max_accu:
            max_accu = test_metrics[0]
            print(f'save train...... \n accuracy before FCL update: {max_accu}, epoch: {epoch}')
            save.save_model_w_condition(model=model, model_dir=model_dir, model_name='warmup_joint_best_val_weights_acc', accu=max_accu, target_accu=0.70, log=log)

        # prototype projection
        if epoch >= cfg.train.push_start and epoch in push_epochs:
            print('Projection: save/push prototypes')
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=multi_model, # pytorch network with prototype_vectors  
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=cfg.paths.img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=cfg.prototype.prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=cfg.prototype.prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=cfg.prototype.proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log,
                epoch=epoch+1)
            
            push_metrics = tnt.test(cfg=cfg, model=multi_model, dataloader=val_loader, class_specific=class_specific, estimator=test_estimator, log=log)

            if push_metrics[0] > max_push_accu:
                max_push_accu = push_metrics[0]
                print(f'save push.. \n inference with visualization proto: {max_push_accu} epoch: {epoch}')
                
                save.save_model_w_condition(model=model, model_dir=model_dir, model_name='push_best_val_weights_acc', accu=max_push_accu, target_accu=0.70, log=log)
            
            if cfg.prototype.activation_function != 'linear':
                tnt.last_only(model=multi_model, cfg=cfg, log=log) # only train the FCL
            
                print('train the last layer')
                log('\n FCL Epoch: \t{0}'.format(epoch))
                for i in range(1, train_fcl_epoch+1):
                    train_estimator.reset() 
                    test_estimator.reset()
                    local_fcl_acc = -1

                    log('iteration: \t{0}'.format(i))
                    train_fcl = tnt.train(cfg=cfg, model=multi_model, dataloader=train_loader, optimizer=last_layer_optimizer, coefs=coefs, 
                                          class_specific=class_specific, estimator=train_estimator, log=log, epoch=i, t_epoch=train_fcl_epoch)

                    test_fcl = tnt.test(cfg=cfg, model=multi_model, dataloader=val_loader, class_specific=class_specific, estimator=test_estimator, log=log, tcurent_epoch=epoch+i)
                    
                    list_train_fcl[0].append(train_fcl[0]*100); list_train_fcl[1].append(train_fcl[1])
                    list_test_fcl[0].append(test_fcl[0]*100); list_test_fcl[1].append(test_fcl[1]*100)
                    list_lnorm[0].append(train_fcl[2]); list_lnorm[1].append(test_fcl[2])

                    logger.add_scalar('train_fcl_acc', train_fcl[0], epoch+i); logger.add_scalar('train_fcl_loss', train_fcl[1], epoch+i)
                    logger.add_scalar('val_fcl_acc', test_fcl[0], epoch+i); logger.add_scalar('val_fcl_loss', test_fcl[1], epoch+i)
                    
                    if (cfg.data.binary) or (cfg.data.threshold):    
                        logger.add_scalar('train_fcl_auc', train_estimator.get_auc_auprc(6)[0][0], epoch+i)
                        list_train_fcl[3].append(train_estimator.get_auc_auprc(6)[0][0])

                        logger.add_scalar('val_fcl_auc', test_estimator.get_auc_auprc(6)[0][0], epoch+i) 
                        list_test_fcl[3].append(test_estimator.get_auc_auprc(6)[0][0])

                        if (cfg.data.binary):
                            kap_auc = test_estimator.get_auc_auprc(6)[0][0]
                            if kap_auc > max_FCL_kappa_auc:                            
                                print(f'save on FCL.. AUC: {max_FCL_kappa_auc}, epoch {epoch} ---> {i}')
                                max_FCL_kappa_auc = kap_auc
                                push_nun_kappa_auc = epoch
                                best_epoch_val_kappa_auc = i
                                save.save_model_w_condition(model=model, model_dir=model_dir, model_name='fcl_best_val_weights_auc', accu=max_FCL_kappa_auc, target_accu=0, log=log)

                    if not cfg.data.binary:
                        kap_auc = test_estimator.get_kappa(6)
                        logger.add_scalar('train_fcl_kappa.', train_estimator.get_kappa(6), epoch+i)
                        list_train_fcl[2].append(kap_auc)

                        logger.add_scalar('val_fcl_kappa.', test_estimator.get_kappa(6), epoch+i)
                        list_test_fcl[2].append(test_estimator.get_kappa(6))

                        if kap_auc >= max_FCL_kappa_auc:
                            print(f'save on FCL.. kappa: {max_FCL_kappa_auc}, epoch {epoch} ---> {i}')
                            max_FCL_kappa_auc = kap_auc
                            push_nun_kappa_auc = epoch
                            best_epoch_val_kappa_auc = i
                            save.save_model_w_condition(model=model, model_dir=model_dir, model_name='fcl_best_val_weights_kappa', accu=max_FCL_kappa_auc, target_accu=0, log=log)

                    if cfg.data.threshold: 
                        logger.add_scalar('train_fcl_bin_acc', train_estimator.get_accuracy(6)[1], epoch+i)  
                        list_train_fcl[4].append(train_estimator.get_accuracy(6)[1])

                        logger.add_scalar('val_fcl_bin_acc', test_estimator.get_accuracy(6)[1], epoch+i) 
                        list_test_fcl[4].append(test_estimator.get_accuracy(6)[1])
                    
                    if test_fcl[0] > max_FCL_accu:
                        print(f'save on FCL.. accuracy: {max_FCL_accu}, epoch {epoch} ---> {i}')
                        max_FCL_accu = test_fcl[0]
                        push_nun = epoch  
                        best_epoch_val_acc = i  
                        save.save_model_w_condition(model=model, model_dir=model_dir, model_name='fcl_best_val_weights_acc', accu=max_FCL_accu, target_accu=0, log=log)   
                    
                    ## save the best local model after projection 
                    if test_fcl[0] > local_fcl_acc:
                        local_path = os.path.join(model_dir, f'img/epoch-{epoch}')
                        local_fcl_acc = test_fcl[0]
                        save.save_model_w_condition(model=model, model_dir=local_path, model_name=f'push_{epoch}_fcl_best_val_weights_acc', accu=local_fcl_acc, target_accu=0, log=log)

                if cfg.data.binary:  
                    df3 = pd.DataFrame({'train_acc': list_train_fcl[0], 'train_loss': list_train_fcl[1],
                                        'train_auc': list_train_fcl[3], 
                                        'val_acc': list_test_fcl[0], 'val_loss': list_test_fcl[1], 
                                        'val_auc': list_test_fcl[3], 
                                        'l1_train': list_lnorm[0], 'l1_test': list_lnorm[1] }) 
                    
                df3.to_csv(os.path.join(paths, 'train_test_fcl.csv'), index=False)

        save.save_model_w_condition(model=model, model_dir=model_dir, model_name='previous_epoch', accu=0.8, target_accu=0, log=log)
        
        df1 = pd.DataFrame({'train_loss': list_train[1], 'train_acc': list_train[0], 'val_acc': list_test[0], 
                            'val_loss': list_test[1]})
        df2 = pd.DataFrame({'push_acc': list_push[0], 'push_loss': list_push[1]})

        df1.to_csv(os.path.join(paths, 'train_test.csv'), index=False)
        df2.to_csv(os.path.join(paths, 'val_push.csv'), index=False)

    print(f'final push number: {push_nun}')
    log('final push number: \t{0}'.format(push_nun))
    log('Best val accuracy: \t{0}, epoch: \t{0}'.format(max_FCL_accu, best_epoch_val_acc))

    log('final push number AUC: \t{0}'.format(push_nun_kappa_auc))
    log('Best val kappa/AUC: \t{0}, epoch: \t{0}'.format(max_FCL_accu, best_epoch_val_kappa_auc))
    save.save_model_w_condition(model=model, model_dir=model_dir, model_name='final_weights', accu=test_fcl[0], target_accu=0.70, log=log)
    
    logclose()
    logger.close() 

#evaluate(cfg, model, checkpoint, val_dataset, estimator, type_ds='validation')
def evaluate(cfg, model_path, test_dataset, estimator, type_ds):
    loss = torch.nn.CrossEntropyLoss()
    model = torch.load(model_path)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        pin_memory=cfg.train.pin_memory
    )

    print(f'Running on {type_ds} set...')
    eval(cfg, model, test_loader, estimator, loss_func=loss)

    #print('========================================')
    if cfg.data.binary:
        list_auc, list_auprc, list_others = estimator.get_auc_auprc(5)
        auc = list_auc[0]
        auprc, sens, prec, spec = list_auprc[0], list_others[0], list_others[1], list_others[2]

        print('Finished! {} acc {}'.format(type_ds, estimator.get_accuracy(6)))
        print('loss:', estimator.get_val_loss())
        print('AUC: {}, sens: {}, spec: {}, prec: {}, AUPRC: {}'.format(auc, sens, spec, prec, auprc))
        print('Confusion Matrix:')
        print(estimator.conf_mat)   
    else:
        print('acc.: {}'.format(estimator.get_accuracy(6)[0]))
        print('binary acc.: {}'.format(estimator.get_accuracy(6)[1]))
        print('kappa: {}'.format(estimator.get_kappa(6)))
        print(estimator.conf_mat)
    #print('========================================')

def eval(cfg, model, dataloader, estimator, loss_func=None):
    model.eval()
    device = cfg.base.device
    criterion = cfg.train.criterion
    torch.set_grad_enabled(False)
    l = {}
    if loss_func:
        loss_function = loss_func       
        l['op'] = lambda tensor: tensor.item()
    else:
        loss_function = lambda a,b: 1
        l['op'] = lambda a: 0

    estimator.reset()
    epoch_loss, avg_val_loss = 0, 0
    for step, test_data in enumerate(dataloader):
        X, y = test_data
        X, y = X.to(device), y.to(device)

        y_pred,_,_ = model(X)
        estimator.update(y_pred, y)

        loss = loss_function(y_pred, y)
        epoch_loss += l['op'](loss) #loss.item()
        avg_val_loss = epoch_loss / (step + 1)

    if loss_func:
        estimator.update_val_loss(avg_val_loss)

    model.train()
    torch.set_grad_enabled(True)