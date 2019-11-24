# How to use Hooker
加入了一个 class Hooker 在 code/utils/hooker.py 中  
## 成员函数
包括__init__ , hooker_insert , remove , save , plot_hist   

## 调用方式
### 初始化及插入hooker
在训练开始阶段(epoch=0或希望开始观测中间数据的某个epoch)，创建Hooker类的对象，参数输入网络模型和希望观测的数据类型('a':activation,'w':weight,'g':gradient),因为__init__函数中也包含hooker_insert函数，因此会自动开始测量。  
而接下来继续使用该Hooker类的对象时，需要在每个epoch都调用一次hook_insert()函数
```
for epoch in range(args.epoch):  

    if epoch == 0:
        a_hooker = Hooker(model = net, vtype = 'a')
        w_hooker = Hooker(model = net, vtype = 'w')
        g_hooker = Hooker(model = net, vtype = 'g')
    else:
        a_hooker.hook_insert()
        w_hooker.hook_insert()
        g_hooker.hook_insert()
```

### remove
插入的hooker必须在适当时候被remove，否则在每次调用forward函数时(每个batch)都会执行hooker的函数，这样会导致记录的数据非常非常多。    
因此可以选择在每个epoch的batch 0结束后就调用remove()，具体实例见下：这里因为每个epoch的训练函数单独编写了一个函数train(),因此需要将这些hooker一起作为参数送到train()函数里调用remove()
```
all_hooker = [a_hooker,w_hooker,g_hooker]
train_loss, train_acc = train(trainloader, net, criterion, optimizer, epoch, args.cuda, all_hooker)
```
train(trainloader, net, criterion, optimizer, epoch, use_cuda, all_hooker)函数内部：
```
for i, data in enumerate(trainloader):
    #start training......
    #end this batch training
    if i == 0:
    for each_hooker in all_hooker:
        each_hooker.remove()
```

### save
save函数的参数包含 (ctype = [], output_path = '', resume = False)   
ctype应是一个列表，内容为以下字符串的组合：'mean','std','max','min','p25','p50','p75'   
output_path为输出文件的地址, resume在epoch=0(第一次调用save时)设为False，其他时刻设置为True 
ctype中选择的计算项即这个Hooker会计算并保存的项 
```
if epoch == 0:
    a_hooker.save(ctype = ['max','min'], output_path = args.output_dir, resume = False)
    w_hooker.save(ctype = ['mean','std'], output_path = args.output_dir, resume = False)
    g_hooker.save(ctype = ['p25','p50'], output_path = args.output_dir, resume = False)
else:
    a_hooker.save(ctype = ['max','min'], output_path = args.output_dir, resume = True)
    w_hooker.save(ctype = ['mean','std'], output_path = args.output_dir, resume = True)
    g_hooker.save(ctype = ['p25','p50'], output_path = args.output_dir, resume = True)
```

### plot_hist
plot_hist用于绘制某一层在某一个epoch(规定为batch 0)时，某个中间值(activation/weight/gradient)的分布图   
参数：name为指定layer的名字(来自用户自己创建网络时的定义，或pytorch自动赋予)。程序开始会打印所有layer的名字，与此处name一致 
output_path:图片保存位置
epoch:当前epoch，用于文件名记录
```
if (epoch == 0) or (epoch == 30) or (epoch == 60) or (epoch == 90) or (epoch == 120):
    a_hooker.plot_hist(name = 'layers.2.1.conv1', output_path = args.output_dir, epoch = epoch)
    w_hooker.plot_hist(name = 'layers.2.2.conv2', output_path = args.output_dir, epoch = epoch)
    g_hooker.plot_hist(name = 'layers.0.1.conv1', output_path = args.output_dir, epoch = epoch)
```

### plot_curve
在整个训练过程结束后，可调用plot_curve函数实现CurvePlot:某一层中的a/w/g随epoch的变化
需保证保存了所有值：'max','min','p25','p50','p75','mean','std'
```
a_hooker.plot(file_path = args.output_dir)
w_hooker.plot(file_path = args.output_dir)
g_hooker.plot(file_path = args.output_dir)
```

### plot_line
调用plot_line函数实现LinePlot:某个epoch不同层的值  
需保证保存了'max','min','p25','p50','p75'  
参数epoch为想要绘制图像的epoch，从1开始计
```
plot_epoch = 2 # index from 1
a_hooker.plot_line(file_path = args.output_dir, epoch = plot_epoch)
w_hooker.plot_line(file_path = args.output_dir, epoch = plot_epoch)
g_hooker.plot_line(file_path = args.output_dir, epoch = plot_epoch)
```
