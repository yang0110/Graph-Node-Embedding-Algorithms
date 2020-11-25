import os.path as osp 
import torch 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style='white')
from sklearn.manifold import TSNE 
from torch_geometric.datasets import Planetoid 
from torch_geometric.nn import Node2Vec, GCNConv
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
output_path = '../graph_data/'
result_path = 'results/'


dataset = Planetoid(root=output_path+'Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = GCNConv(dataset.num_features, 16)
		self.conv2 = GCNConv(16, dataset.num_classes)

	def forward(self):
		x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr 
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index, edge_weight)
		return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
	model.train()
	out = model()[data.train_mask]
	loss = F.nll_loss(out, data.y[data.train_mask])
	loss.backward()
	optimizer.step()
	return loss.item()

@torch.no_grad()
def test():
	model.eval()
	out = model()
	accs = []
	pred = out[data.test_mask].max(1)[1]
	acc = pred.eq(data.y[data.test_mask]).sum().item()/data.test_mask.sum().item()
	return acc 


epoch_num = 100
loss_list = []
acc_list = []

for epoch in range(epoch_num):
	loss = train()
	acc = test()
	loss_list.append(loss)
	acc_list.append(acc)
	print('epoch %s, loss %.2f, acc %.2f'%(epoch, loss, acc))

plt.figure(figsize=(6,4))
plt.plot(loss_list)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'gcn_loss'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(acc_list)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('acc', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'gcn_acc'+'.png', dpi=100)
plt.show()









