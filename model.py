import torch
import torch.nn as nn

def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    if type(m) == torch.nn.Linear:
        init_fn(m.weight) #初始化权重


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    ) #fc_block就是一个全连接层加上一个relu


class OccupancyMap(torch.nn.Module):  #计算体密度和颜色的网络
    def __init__(
        self,
        emb_size1,
        emb_size2,
        hidden_size=256,
        do_color=True,
        hidden_layers_block=1
    ):
        super(OccupancyMap, self).__init__()
        self.do_color = do_color #是否计算颜色
        self.embedding_size1 = emb_size1
        self.in_layer = fc_block(self.embedding_size1, hidden_size) #输入层，输入embedding_size1，输出是hidden_size

        hidden1 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)] #中间层，输入是hidden_size，输出是hidden_size
        self.mid1 = torch.nn.Sequential(*hidden1) #mid1是中间层hidden1的组合
        # self.embedding_size2 = 21*(5+1)+3 - self.embedding_size # 129-66=63 32
        self.embedding_size2 = emb_size2
        self.cat_layer = fc_block(
            hidden_size + self.embedding_size1, hidden_size) #cat_layer的输入是hidden_size + self.embedding_size1，输出是hidden_size

        # self.cat_layer = fc_block(
        #     hidden_size , hidden_size)

        hidden2 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)] #中间层hidden2，输入是hidden_size，输出是hidden_size
        self.mid2 = torch.nn.Sequential(*hidden2) #mid2是中间层hidden2的组合

        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        if self.do_color:
            self.color_linear = fc_block(self.embedding_size2 + hidden_size, hidden_size) #color_linear的输入是self.embedding_size2 + hidden_size，输出是hidden_size
            self.out_color = torch.nn.Linear(hidden_size, 3) #out_color的输入是hidden_size，输出是3,即rgb颜色

        # self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x,
                noise_std=None,
                do_alpha=True,
                do_color=True,
                do_cat=True):
        fc1 = self.in_layer(x[...,:self.embedding_size1]) #输入层，输入是x[...,:self.embedding_size1]，输出是hidden_size
        fc2 = self.mid1(fc1) #中间层，输入hidden_size，输出是hidden_size
        # fc3 = self.cat_layer(fc2)
        if do_cat:
            fc2_x = torch.cat((fc2, x[...,:self.embedding_size1]), dim=-1) #将fc2和x[...,:self.embedding_size1]拼接起来
            fc3 = self.cat_layer(fc2_x) #输入是hidden_size + x[...,:self.embedding_size1]，输出是hidden_size
        else:
            fc3 = fc2
        fc4 = self.mid2(fc3) #mid2，输入是hidden_size，输出是hidden_size

        alpha = None
        if do_alpha: #计算体密度
            raw = self.out_alpha(fc4)   # todo ignore noise,输入是hidden_size，输出是1，即体密度
            if noise_std is not None:
                noise = torch.randn(raw.shape, device=x.device) * noise_std
                raw = raw + noise #给体密度加上噪声，为什么加噪声？是因为噪声可以使得体密度更加平滑，更加真实

            # alpha = self.relu(raw) * scale    # nerf
            alpha = raw * 10. #self.scale     # unisurf #体密度的缩放倍率self.scale是10

        color = None
        if self.do_color and do_color:
            fc4_cat = self.color_linear(torch.cat((fc4, x[..., self.embedding_size1:]), dim=-1)) #将fc4和x[..., self.embedding_size1:]拼接起来，输入是hidden_size + x[..., self.embedding_size1:]，输出是hidden_size
            raw_color = self.out_color(fc4_cat) #输入是hidden_size，输出是3，即rgb颜色
            color = self.sigmoid(raw_color) #最后通过sigmoid函数，将rgb颜色的值限制在0到1之间

        return alpha, color



