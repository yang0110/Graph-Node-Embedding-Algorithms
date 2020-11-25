import os.path as osp 
import torch 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style='white')
from sklearn.manifold import TSNE 
from torch_geometric.datasets import Planetoid 
from torch_geometric.nn import Node2Vec 
from torch_geometric.data import DataLoader
output_path = '../graph_data/'
result_path = 'results/'

print('load data')
dataset = Planetoid(root=output_path+'Cora', name='Cora')
data = dataset[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('create model')
model = Node2Vec(data.edge_index, embedding_dim=20, walk_length=10, context_size=2, walks_per_node=5, num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = model.loader(batch_size=32, shuffle=True, num_workers=0)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask], z[data.test_mask], data.y[data.test_mask], max_iter=50)
    return acc 

print('train model')
loss_list = []
acc_list = []
epoch_num = 100
for epoch in range(epoch_num):
    loss = train()
    acc = test()
    loss_list.append(loss)
    acc_list.append(acc)
    print('epoch %s, loss %.2f, acc %.2f'%(epoch, loss, acc))


@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(6,4))
    for i in range(dataset.num_classes):
        plt.scatter(z[y== i, 0], z[y==i, 1], s=20, color=colors[i])

    plt.axis('off')
    plt.show()

# colors = ['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700']

# plot_points(colors)


plt.figure(figsize=(6,4))
plt.plot(loss_list)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'node2vec_loss'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(acc_list)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'node2vec_acc'+'.png', dpi=100)
plt.show()


















