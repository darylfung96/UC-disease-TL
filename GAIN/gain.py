#%% Packages
import torch
import numpy as np
import os
from tqdm import tqdm
# from tqdm.notebook import tqdm_notebook as tqdm
import torch.nn.functional as F
import pytorch_lightning as pl

from data_preprocessing import dataset_list


# %% Necessary Functions

# 1. Xavier Initialization Definition
# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape = size, stddev = xavier_stddev)
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev)


# Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C


# %% 3. Other functions
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size=[m, n])


# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def test_loss(X, M, New_X):
    # %% Structure
    # Generator
    G_sample = generator(New_X, M)

    # %% MSE Performance metric
    MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
    return MSE_test_loss, G_sample


class GAIN(pl.LightningModule):
    def __init__(self, dataset):
        super(GAIN, self).__init__()
        self.gain_checkpoint_dir = 'gain_checkpoints'
        os.makedirs(self.gain_checkpoint_dir, exist_ok=True)

        current_dataset = dataset_list[dataset](dataset)

        output_dict = current_dataset.process_data(pad_in_sequence=True)
        sorted_data, sorted_length, target_data, missing_data = output_dict['sorted_data'], output_dict[
            'sorted_length'], \
                                                                output_dict['target_data'], output_dict['missing_data']
        sorted_data = sorted_data.reshape(-1, sorted_data.shape[-1])
        missing_data = missing_data.reshape(-1, missing_data.shape[-1])

        dataset_file = 'Spam.csv'  # 'Letter.csv' for Letter dataset an 'Spam.csv' for Spam dataset

        self.use_gpu = torch.cuda.is_available()  # set it to True to use GPU and False to use CPU
        if self.use_gpu:
            torch.cuda.set_device(0)

        # %% System Parameters
        # 1. Mini batch size
        self.mb_size = 64
        # 2. Missing rate
        self.p_miss = 0.2
        # 3. Hint rate
        self.p_hint = 0.9
        # 4. Loss Hyperparameters
        self.alpha = 10
        # 5. Train Rate
        train_rate = 0.8

        # %% Data

        # Data generation
        # Data = np.loadtxt(dataset_file, delimiter=",", skiprows=1)
        Data = sorted_data
        # Parameters
        No = len(Data)
        self.Dim = len(Data[0, :])

        # Hidden state dimensions
        self.H_Dim1 = self.Dim
        self.H_Dim2 = self.Dim

        # Normalization (0 to 1)
        Min_Val = np.zeros(self.Dim)
        Max_Val = np.zeros(self.Dim)

        for i in range(self.Dim):
            Min_Val[i] = np.min(Data[:, i])
            Data[:, i] = Data[:, i] - np.min(Data[:, i])
            Max_Val[i] = np.max(Data[:, i])
            Data[:, i] = Data[:, i] / (np.max(Data[:, i]) + 1e-6)

        # %% Missing introducing
        # p_miss_vec = p_miss * np.ones((Dim, 1))
        #
        # Missing = np.zeros((No, Dim))
        #
        # for i in range(Dim):
        #     A = np.random.uniform(0., 1., size=[len(Data), ])
        #     B = A > p_miss_vec[i]
        #     Missing[:, i] = 1. * B
        Missing = 1 - missing_data

        # %% Train Test Division

        idx = np.random.permutation(No)

        self.Train_No = int(No * train_rate)
        Test_No = No - self.Train_No

        # Train / Test Features
        self.trainX = Data[idx[:self.Train_No], :]
        self.testX = Data[idx[self.Train_No:], :]

        # Train / Test Missing Indicators
        self.trainM = Missing[idx[:self.Train_No], :]
        self.testM = Missing[idx[self.Train_No:], :]

        self.theta_G, self.theta_D = self._initialize_weight()

    def _initialize_weight(self):
        # %% 1. Discriminator
        if self.use_gpu is True:
            self.D_W1 = torch.nn.Parameter(torch.tensor(xavier_init([self.Dim * 2, self.H_Dim1]), requires_grad=True,
                                device="cuda"))  # Data + Hint as inputs
            self.D_b1 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.H_Dim1]), requires_grad=True, device="cuda"))

            self.D_W2 = torch.nn.Parameter(torch.tensor(xavier_init([self.H_Dim1, self.H_Dim2]), requires_grad=True, device="cuda"))
            self.D_b2 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.H_Dim2]), requires_grad=True, device="cuda"))

            self.D_W3 = torch.nn.Parameter(torch.tensor(xavier_init([self.H_Dim2, self.Dim]), requires_grad=True, device="cuda"))
            self.D_b3 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.Dim]), requires_grad=True, device="cuda"))  # Output is multi-variate
        else:
            self.D_W1 = torch.nn.Parameter(torch.tensor(xavier_init([self.Dim * 2, self.H_Dim1]), requires_grad=True))  # Data + Hint as inputs
            self.D_b1 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.H_Dim1]), requires_grad=True))

            self.D_W2 = torch.nn.Parameter(torch.tensor(xavier_init([self.H_Dim1, self.H_Dim2]), requires_grad=True))
            self.D_b2 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.H_Dim2]), requires_grad=True))

            self.D_W3 = torch.nn.Parameter(torch.tensor(xavier_init([self.H_Dim2, self.Dim]), requires_grad=True))
            self.D_b3 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.Dim]), requires_grad=True))  # Output is multi-variate

        theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

        # %% 2. Generator
        if self.use_gpu is True:
            self.G_W1 = torch.nn.Parameter(torch.tensor(xavier_init([self.Dim * 2, self.H_Dim1]), requires_grad=True,
                                device="cuda"))  # Data + Mask as inputs (Random Noises are in Missing Components
            self.G_b1 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.H_Dim1]), requires_grad=True, device="cuda"))

            self.G_W2 = torch.nn.Parameter(torch.tensor(xavier_init([self.H_Dim1, self.H_Dim2]), requires_grad=True, device="cuda"))
            self.G_b2 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.H_Dim2]), requires_grad=True, device="cuda"))

            self.G_W3 = torch.nn.Parameter(torch.tensor(xavier_init([self.H_Dim2, self.Dim]), requires_grad=True, device="cuda"))
            self.G_b3 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.Dim]), requires_grad=True, device="cuda"))
        else:
            self.G_W1 = torch.nn.Parameter(torch.tensor(xavier_init([self.Dim * 2, self.H_Dim1]),
                                requires_grad=True))  # Data + Mask as inputs (Random Noises are in Missing Components)
            self.G_b1 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.H_Dim1]), requires_grad=True))

            self.G_W2 = torch.nn.Parameter(torch.tensor(xavier_init([self.H_Dim1, self.H_Dim2]), requires_grad=True))
            self.G_b2 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.H_Dim2]), requires_grad=True))

            self.G_W3 = torch.nn.Parameter(torch.tensor(xavier_init([self.H_Dim2, self.Dim]), requires_grad=True))
            self.G_b3 = torch.nn.Parameter(torch.tensor(np.zeros(shape=[self.Dim]), requires_grad=True))

        theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

        return theta_G, theta_D

    # %% 1. Generator
    def _generator(self, new_x, m):
        inputs = torch.cat(dim=1, tensors=[new_x, m])  # Mask + Data Concatenate
        G_h1 = F.relu(torch.matmul(inputs, self.G_W1) + self.G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, self.G_W2) + self.G_b2)
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.G_W3) + self.G_b3)  # [0,1] normalized Output

        return G_prob

    # %% 2. Discriminator
    def _discriminator(self, new_x, h):
        inputs = torch.cat(dim=1, tensors=[new_x, h])  # Hint + Data Concatenate
        D_h1 = F.relu(torch.matmul(inputs, self.D_W1) + self.D_b1)
        D_h2 = F.relu(torch.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = torch.matmul(D_h2, self.D_W3) + self.D_b3
        D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
        return D_prob

    def _discriminator_loss(self, M, New_X, H):
        # Generator
        G_sample = self._generator(New_X, M)
        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = self._discriminator(Hat_New_X, H)

        # %% Loss
        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob + 1e-8))
        return D_loss

    def _generator_loss(self, X, M, New_X, H):
        # %% Structure
        # Generator
        G_sample = self._generator(New_X, M)

        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = self._discriminator(Hat_New_X, H)

        # %% Loss
        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
        MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)

        G_loss = G_loss1 + self.alpha * MSE_train_loss

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        return G_loss, MSE_train_loss, MSE_test_loss

    def generate(self, X, m):
        if type(X) != torch.Tensor:
            X = torch.DoubleTensor(X)
        if type(m) != torch.Tensor:
            m = torch.DoubleTensor(m)
        return self._generator(X, m)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # %% Inputs
        mb_idx = sample_idx(self.Train_No, self.mb_size)
        X_mb = self.trainX[mb_idx, :]

        Z_mb = sample_Z(self.mb_size, self.Dim)
        M_mb = self.trainM[mb_idx, :]
        H_mb1 = sample_M(self.mb_size, self.Dim, 1 - self.p_hint)
        H_mb = M_mb * H_mb1

        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

        if self.use_gpu is True:
            X_mb = torch.tensor(X_mb, device="cuda")
            M_mb = torch.tensor(M_mb, device="cuda")
            H_mb = torch.tensor(H_mb, device="cuda")
            New_X_mb = torch.tensor(New_X_mb, device="cuda")
        else:
            X_mb = torch.tensor(X_mb)
            M_mb = torch.tensor(M_mb)
            H_mb = torch.tensor(H_mb)
            New_X_mb = torch.tensor(New_X_mb)

        if optimizer_idx == 0:
            G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = self._generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb,
                                                                                  H=H_mb)
            self.log('generator training loss', G_loss_curr.item(), prog_bar=True)
            return G_loss_curr

        if optimizer_idx == 1:
            D_loss_curr = self._discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
            self.log('discriminator training loss', D_loss_curr.item(), prog_bar=True)
            return D_loss_curr

    def save(self):
        torch.save(self.state_dict(), os.path.join(self.gain_checkpoint_dir, 'best.ckpt'))

    def load(self, checkpoint_name='best.ckpt'):
        loaded_state_dict = torch.load(os.path.join(self.gain_checkpoint_dir, checkpoint_name), map_location=torch.device('cpu'))
        self.load_state_dict(loaded_state_dict)

    def on_epoch_end(self):
        if self.current_epoch % 5 == 0:
            self.save()

    def configure_optimizers(
            self,
    ):
        optimizer_G = torch.optim.Adam(params=self.theta_G)
        optimizer_D = torch.optim.Adam(params=self.theta_D)

        return [optimizer_G, optimizer_D], []


# %% Start Iterations
# for it in tqdm(range(5000)):
#
#     # %% Inputs
#     mb_idx = sample_idx(Train_No, mb_size)
#     X_mb = trainX[mb_idx, :]
#
#     Z_mb = sample_Z(mb_size, Dim)
#     M_mb = trainM[mb_idx, :]
#     H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
#     H_mb = M_mb * H_mb1
#
#     New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce
#
#     if use_gpu is True:
#         X_mb = torch.tensor(X_mb, device="cuda")
#         M_mb = torch.tensor(M_mb, device="cuda")
#         H_mb = torch.tensor(H_mb, device="cuda")
#         New_X_mb = torch.tensor(New_X_mb, device="cuda")
#     else:
#         X_mb = torch.tensor(X_mb)
#         M_mb = torch.tensor(M_mb)
#         H_mb = torch.tensor(H_mb)
#         New_X_mb = torch.tensor(New_X_mb)
#
#     optimizer_D.zero_grad()
#     D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
#     D_loss_curr.backward()
#     optimizer_D.step()
#
#     optimizer_G.zero_grad()
#     G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
#     G_loss_curr.backward()
#     optimizer_G.step()
#
#     # %% Intermediate Losses
#     if it % 100 == 0:
#         print('Iter: {}'.format(it), end='\t')
#         print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())), end='\t')
#         print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
