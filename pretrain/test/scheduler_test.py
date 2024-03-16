import torch
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

SD_CLASS = [
    # 'StepLR', 
    # 'LinearLR', 
    # 'CosineAnnealingLR', 
    'OneCycleLR', 
]

num_epochs = 20
n_batches = 100
T_max = num_epochs * n_batches
learning_rate = 3e-4

for scheduler_type in SD_CLASS:
    
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    if scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=n_batches, gamma=0.5)
    elif scheduler_type == 'LinearLR':
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=100)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate, 
            steps_per_epoch=n_batches, 
            pct_start=0.08,
            epochs=num_epochs, 
            anneal_strategy='linear'
        )
    else:
        raise ValueError(f"Invalid scheduler_type: {scheduler_type}")
    
    print(f"Scheduler: {scheduler_type}")

    lr_list = []
    for epoch in range(num_epochs):
        for i in range(n_batches):
            # model eval

            # model train

            optimizer.zero_grad()
            # loss.backward()
            optimizer.step()
            lr_list.append(scheduler.get_last_lr())
            scheduler.step()

    plt.figure(figsize=(10, 5))
    plt.plot(lr_list)
    plt.title(scheduler_type)
    plt.savefig(f"images/{scheduler_type}.png")
    

    


        