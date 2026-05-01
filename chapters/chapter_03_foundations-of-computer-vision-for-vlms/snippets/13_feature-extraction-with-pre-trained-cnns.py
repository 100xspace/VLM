5.  model = models.resnet50(pretrained=True)
6.  model = torch.nn.Sequential(*(list(model.children())[:-1]))
7.  model.eval()
