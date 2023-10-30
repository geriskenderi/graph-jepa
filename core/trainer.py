import os
import torch
import random
import time
import numpy as np
from core.log import config_logger
from core.asam import ASAM
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(cfg, create_dataset, create_model, train, test, evaluator=None):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [21, 42, 41, 95, 12, 35, 66, 85, 3, 1234]

    writer, logger = config_logger(cfg)

    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    train_loader = DataLoader(
        train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(
        val_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(
        test_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

    train_losses = []
    per_epoch_times = []
    total_times = []
    maes = []
    for run in range(cfg.train.runs):
        set_seed(seeds[run])
        model = create_model(cfg).to(cfg.device)
        print(f"\nNumber of parameters: {count_parameters(model)}")

        if cfg.train.optimizer == 'ASAM':
            sharp = True
            optimizer = torch.optim.SGD(
                model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.wd)
            minimizer = ASAM(optimizer, model, rho=0.5)

        else:
            sharp = False
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=cfg.train.lr_decay,
                                                               patience=cfg.train.lr_patience,
                                                               verbose=True)

        start_outer = time.time()
        per_epoch_time = []
        
        # Create EMA scheduler for target encoder param update
        ipe = len(train_loader)
        ema_params = [0.996, 1.0]
        momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.train.epochs)
                            for i in range(int(ipe*cfg.train.epochs)+1))
        for epoch in range(cfg.train.epochs):
            start = time.time()
            model.train()
            _, train_loss = train(
                train_loader, model, optimizer if not sharp else minimizer, \
                    evaluator=evaluator, device=cfg.device, momentum_weight=next(momentum_scheduler),\
                    sharp=sharp, criterion_type=cfg.jepa.dist)
            model.eval()
            _, val_loss = test(val_loader, model,
                                      evaluator=evaluator, device=cfg.device, criterion_type=cfg.jepa.dist)
            _, test_loss = test(test_loader, model,
                                      evaluator=evaluator, device=cfg.device, criterion_type=cfg.jepa.dist)

            time_cur_epoch = time.time() - start
            per_epoch_time.append(time_cur_epoch)

            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f} Seconds: {time_cur_epoch:.4f}')

            writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Run{run}/val-loss', val_loss, epoch)

            if scheduler is not None:
                scheduler.step(val_loss)

            if not sharp:
                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print("!! LR EQUAL TO MIN LR SET.")
                    break

            # if cfg.dataset in ['TreeDataset', 'sr25-classify'] and test_perf == 1.0:
            #     break
            # torch.cuda.empty_cache()  # empty test part memory cost

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time()-start_outer)/3600

        model.eval()
        X_train, y_train = [], []
        X_test, y_test = [], []
        ### Extracting training features and labels in Scikit-Learn form
        for data in train_loader:
            data.to(cfg.device)
            with torch.no_grad():
                features = model.encode(data)
                X_train.append(features.detach().cpu().numpy())
                y_train.append(data.y.detach().cpu().numpy())

        # Concatenate the lists into numpy arrays
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        for data in test_loader:
            data.to(cfg.device)
            with torch.no_grad():
                features = model.encode(data)
                X_test.append(features.detach().cpu().numpy())
                y_test.append(data.y.detach().cpu().numpy())

        # Concatenate the lists into numpy arrays
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        print("Data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # Fine tuning on the learned representations via Ridge Regression
        lin_model = Ridge()
        lin_model.fit(X_train, y_train)
        lin_predictions = lin_model.predict(X_test)
        lin_mae = mean_absolute_error(y_test, lin_predictions)
        maes.append(lin_mae)

        print("\nRun: ", run)
        print("Train Loss: {:.4f}".format(train_loss))
        print("Convergence Time (Epochs): {}".format(epoch+1))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h".format(total_time))
        print(f'Train R2.: {lin_model.score(X_train, y_train)}')
        print(f'MAE.: {lin_mae}')

        train_losses.append(train_loss)
        per_epoch_times.append(per_epoch_time)
        total_times.append(total_time)

    if cfg.train.runs > 1:
        train_loss = torch.tensor(train_losses)
        per_epoch_time = torch.tensor(per_epoch_times)
        total_time = torch.tensor(total_times)
        print(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
              f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
              f'\nHours/total: {total_time.mean():.4f}')
        logger.info("-"*50)
        logger.info(cfg)
        logger.info(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
                    f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
                    f'\nHours/total: {total_time.mean():.4f}')
        maes = np.array(maes)
        print(f'MAE avg: {maes.mean()}, std: {maes.std()}')

def count_parameters(model):
    # For counting number of parameteres: need to remove unnecessary DiscreteEncoder, and other additional unused params
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def k_fold(dataset, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    train_indices, test_indices = [], []
    ys = dataset.data.y
    for train, test in skf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.from_numpy(train).to(torch.long))
        test_indices.append(torch.from_numpy(test).to(torch.long))
    return train_indices, test_indices


def run_k_fold(cfg, create_dataset, create_model, train, test, evaluator=None, k=10):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [42, 21, 95, 12, 35]

    writer, logger = config_logger(cfg)
    dataset, transform, transform_eval = create_dataset(cfg)

    if hasattr(dataset, 'train_indices'):
        k_fold_indices = dataset.train_indices, dataset.test_indices
    else:
        k_fold_indices = k_fold(dataset, cfg.k)

    train_losses = []
    per_epoch_times = []
    total_times = []
    run_metrics = []
    for run in range(cfg.train.runs):
        set_seed(seeds[run])
        acc = []
        for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
            train_dataset.transform = transform
            test_dataset.transform = transform_eval
            test_dataset = [x for x in test_dataset]

            if not cfg.metis.online:
                train_dataset = [x for x in train_dataset]

            train_loader = DataLoader(
                train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
            test_loader = DataLoader(
                test_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

            model = create_model(cfg).to(cfg.device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=cfg.train.lr_decay,
                                                                   patience=cfg.train.lr_patience,
                                                                   verbose=True)

            start_outer = time.time()
            per_epoch_time = []
            

            # Create EMA scheduler for target encoder param update
            ipe = len(train_loader)
            ema_params = [0.996, 1.0]
            momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.train.epochs)
                                for i in range(int(ipe*cfg.train.epochs)+1))
            
            for epoch in range(cfg.train.epochs):
                start = time.time()
                model.train()
                _, train_loss = train(
                    train_loader, model, optimizer, 
                    evaluator=evaluator, device=cfg.device, 
                    momentum_weight=next(momentum_scheduler), criterion_type=cfg.jepa.dist)
                model.eval()
                _, test_loss = test(
                    test_loader, model, evaluator=evaluator, device=cfg.device, 
                    criterion_type=cfg.jepa.dist)

                scheduler.step(test_loss)
                time_cur_epoch = time.time() - start
                per_epoch_time.append(time_cur_epoch)

                print(f'Epoch/Fold: {epoch:03d}/{fold}, Train Loss: {train_loss:.4f}'
                      f' Test Loss:{test_loss:.4f}, Seconds: {time_cur_epoch:.4f}, ')
                writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
                writer.add_scalar(f'Run{run}/test-loss', test_loss, epoch)

                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print("!! LR EQUAL TO MIN LR SET.")
                    break

            per_epoch_time = np.mean(per_epoch_time)
            total_time = (time.time()-start_outer)/3600

           
            # Finetune using a linear and a nonlinear model after training (use scikit-learn for both, easier to implement)
            # Extract data from the dataLoaders
            model.eval()
            X_train, y_train = [], []
            X_test, y_test = [], []

            # Extracting training features and labels in Scikit-Learn api
            for data in train_loader:
                data.to(cfg.device)
                with torch.no_grad():
                    features = model.encode(data)
                    X_train.append(features.detach().cpu().numpy())
                    y_train.append(data.y.detach().cpu().numpy())

            # Concatenate the lists into numpy arrays
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)

            for data in test_loader:
                data.to(cfg.device)
                with torch.no_grad():
                    features = model.encode(data)
                    X_test.append(features.detach().cpu().numpy())
                    y_test.append(data.y.detach().cpu().numpy())

            # Concatenate the lists into numpy arrays
            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            print("Data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            # 1) L2 penalized logistic regression for fine tuning
            lin_model = LogisticRegression(max_iter=10000)
            lin_model.fit(X_train, y_train)
            lin_predictions = lin_model.predict(X_test)
            lin_accuracy = accuracy_score(y_test, lin_predictions)
            acc.append(lin_accuracy)

            print(f'Fold {fold}, Seconds/epoch: {per_epoch_time}')
            print(f'Acc.: {lin_accuracy}')
            train_losses.append(train_loss)
            per_epoch_times.append(per_epoch_time)
            total_times.append(total_time)

        print("\nRun: ", run)
        print("Train Loss: {:.4f}".format(train_loss))
        print("Convergence Time (Epochs): {}".format(epoch+1))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h".format(total_time))
        acc = np.array(acc)
        print(f'Acc mean: {acc.mean()}, std: {acc.std()}')
        run_metrics.append([acc.mean(), acc.std()])
        print()

    if cfg.train.runs > 1:
        train_loss = torch.tensor(train_losses)
        per_epoch_time = torch.tensor(per_epoch_times)
        total_time = torch.tensor(total_times)

        print(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
              f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
              f'\nHours/total: {total_time.mean():.4f}')
        logger.info("-"*50)
        logger.info(cfg)
        logger.info(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
                    f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
                    f'\nHours/total: {total_time.mean():.4f}')

    run_metrics = np.array(run_metrics)
    print('Averages over 5 runs:')
    print(run_metrics[:, 0].mean(), run_metrics[:, 1].mean())
    print()
