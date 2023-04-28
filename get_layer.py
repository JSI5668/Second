import torch
import timm
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
# model = timm.create_model('efficientnet_b5', pretrained=True)
PATH_1 = 'E:/Second_paper/Checkpoint/Camvid/EdgeNet_2_model/model.pt'
model = torch.load(PATH_1)

train_nodes, eval_nodes = get_graph_node_names(model)
print(model)
print(train_nodes)
print(eval_nodes)

return_nodes={
    train_nodes[3]: 'f1',
    train_nodes[6]: 'f2',
    train_nodes[7]: 'f3'
}


# return_nodes={
#     train_nodes[41]: 'f1',
#     train_nodes[115]: 'f2'
# }
aa = create_feature_extractor(model,return_nodes)
print(aa)
inputs = torch.ones(size=(1,3,224,224))

ooo = aa(inputs)
print('ccccc')

aa['f1']
ooo['f1']

##https://pytorch.org/vision/0.11/feature_extraction.html?highlight=extractor