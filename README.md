# PyTorch Template

本项目是深度学习模型pytorch框架的一个通用训练模板，源自 https://github.com/moemen95/Pytorch-Project-Template.

在原项目上按照自己的习惯做了一些修改。

### 文件夹组织机构说明如下：

![](https://cdn.jsdelivr.net/gh/whuhit/Pytorch_Template/data/assets/diagram.png)



### Mnist分类模型示例

`
python main.py
`

训练过程中可以看到保存的模型，文件在experiments/mnist_exp_0/checkpoints下面。

如果想查看模型的效果，可以运行下面的代码。

    agent = MnistAgent(config)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True, num_workers=1)

    for data, target in test_loader:
        out = agent.inference(data)
        pre = out.max(1, keepdim=True)
        print(pre[1][0].item(), target.item())




