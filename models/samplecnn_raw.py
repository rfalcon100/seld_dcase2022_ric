import torch
import torch.nn as nn
import numpy as np
if __name__ == "__main__":
    from conformer import ConformerBlock
else:
    from models.conformer import ConformerBlock


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class SampleCNN(nn.Module):
    def __init__(self, channels_in=4, channels_out=3, dropout=0.5, num_class=12, multi_track=False, output_timesteps=60):
        super(SampleCNN, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.dropout = dropout
        self.num_class = num_class
        self.multi_track = multi_track
        self.output_steps = output_timesteps

        # 144000 x 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.channels_in, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 48000 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 16000 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 5333 x 256
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 1777 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 592 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(self.dropout))
        # 197 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))

        # output: 65 x 128  (65 timesteps, 128 channels)
        self.avgpool = nn.AdaptiveAvgPool1d(self.output_steps)

        self.conformer1 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )
        self.conformer2 = nn.Sequential(
            ConformerBlock(dim=128, dim_head=64)
        )
        self.doa = nn.Sequential(
            TimeDistributed(nn.Linear(128, 128), batch_first=True),
            nn.Dropout(self.dropout),
            TimeDistributed(nn.Linear(128, 3 * self.num_class), batch_first=True),
            #nn.Tanh()
        )
        # output is 60 x 36, timesteps, 3 x class_num

    def forward(self, x):

        #x = x.view(x.shape[0], 8, -1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.avgpool(out)
        b, c, t = out.shape
        out = out.permute(0, 2, 1)
        # out, h = self.rnn1(out)
        # out, h = self.rnn2(out)
        out = self.conformer1(out)
        out = self.conformer2(out)

        out = self.doa(out)
        out = out.permute(0,2,1)  # Back to [batch, channels, timesteps]

        if self.multi_track:
            out = out.view(-1, 3, self.channels_out, self.num_class, t)
        else:
            out = out.view(-1, self.channels_out, self.num_class, t)

        return out


class SampleCNN_GRU(nn.Module):
    def __init__(self, channels_in=4, channels_out=3, dropout=0.5, num_class=12, multi_track=False, output_timesteps=60):
        super(SampleCNN_GRU, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.dropout = dropout
        self.num_class = num_class
        self.multi_track = multi_track
        self.output_steps = output_timesteps

        # 144000 x 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.channels_in, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 48000 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 16000 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 5333 x 256
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 1777 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 592 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(self.dropout))
        # 197 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))

        # output: 65 x 128  (65 timesteps, 128 channels)
        self.avgpool = nn.AdaptiveAvgPool1d(self.output_steps)

        # input_size : the number of expected features
        # hidden_size : hidden features in the gru module
        # input should be : (b, seq_len, H_in)
        self.gru = nn.GRU(input_size=128, hidden_size=256,
                          num_layers=2, batch_first=True, bidirectional=True)

        self.event_fc = nn.Linear(512, self.num_class, bias=True)
        if self.multi_track:
            self.xyz_fc = nn.Linear(512, 3 * 3 * self.num_class, bias=True)
            #self.xyz_fc = nn.Linear(512, 6 * 4 * class_num, bias=True) # mACCDOA with ADPIT, 6 tracks, 4 channels = activity +xyz
        else:
            self.xyz_fc = nn.Linear(512, 3 * self.num_class, bias=True)

    def forward(self, x):
        #x = x.view(x.shape[0], 8, -1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.avgpool(out)
        b, c, t = out.shape
        out = out.permute(0, 2, 1)

        # x = [2, 8, 512]
        #self.gru.flatten_parameters()  # Maybe this is neede when using DataParallel?
        (x, _) = self.gru(out)

        # x.shape = [2, 8, 512]
        if self.channels_out == 1:
            event_output = self.event_fc(x)
            if self.sigmoid:
                event_output = torch.sigmoid(event_output)
            '''(batch_size, time_steps, 1 * class_num)'''
        elif self.channels_out > 1:
            event_output = self.xyz_fc(x)
            '''(batch_size, time_steps, 3 * class_num)'''
            # or (batch_size, time_steps, 4 * 6 * class_num)

        event_output = event_output.transpose(1, 2)
        if self.multi_track:
            event_output = event_output.view(-1, 3, self.channels_out, self.class_num, t)

            b, ch, tra, cls, ts = event_output.shape
            event_output = event_output.permute([0, 4, 1, 2, 3])
            event_output = event_output.view([-1, ts, ch * tra * cls])
        else:
            event_output = event_output.view(-1, self.channels_out, self.num_class, t)

        return event_output



def unit_test_samplecnn_raw():
    """ Quick test for the Samplecnn model"""
    import torch.optim as optim
    from torch.utils.data import DataLoader
    #from torchsummary import summary
    from torchinfo import summary

    learning_rate = 0.0001
    datapoints, batch, epochs = 2, 2, 20000
    input_shape = [datapoints, 4, 144000]
    output_shape = [datapoints, 3, 12, 60]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    x = torch.randn(input_shape).to(device)
    #y = torch.randn(output_shape).to(device)
    y = torch.randn([datapoints, 3, 12, 6]).to(device)
    y = torch.repeat_interleave(y, 10, dim=-1).to(device)
    assert torch.equal(torch.Tensor(output_shape), torch.Tensor(list(y.shape))), 'Wrong shape for outputs.'

    data = torch.utils.data.TensorDataset(x, y)
    dataloader = DataLoader(data, batch_size=batch)
    model = SampleCNN(output_timesteps=output_shape[-1]).to(device)
    #model = SampleCNN_GRU(output_timesteps=output_shape[-1]).to(device)

    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #summary(model, input_size=tuple(input_shape[-2:]))
    summary(model, input_size=tuple(input_shape), col_names=['input_size', 'output_size', 'kernel_size', 'num_params'],
            row_settings=['var_names'])

    model.train()
    for epoch in range(epochs):
        for ctr, (x, target) in enumerate(dataloader):
            # x, target = x.to(device), target.to(device)
            model.zero_grad()
            out = model(x)

            loss = loss_f(out, target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('Epoch: {} / {} , loss {:.8f}'.format(epoch, epochs, loss.item()))
                # print('outputs : {}'.format(out.detach().cpu().numpy()))

    model.eval()
    out = model(x)
    print('')
    print('outputs : {}'.format(out.detach().cpu().numpy()))
    a = out.detach().cpu().numpy()
    b = target.detach().cpu().numpy()
    print('target : {}'.format(b))

    print('')
    print('Beginning tests:')
    assert np.allclose(a, b, atol=1), 'Wrong outputs'
    assert np.allclose(a, b, atol=1.e-1), 'Wrong outputs'
    assert np.allclose(a, b, atol=1.e-2), 'Wrong outputs'
    assert np.allclose(a, b, atol=1.e-3), 'Wrong outputs'

    print('Unit test completed.')


if __name__ == '__main__':
    unit_test_samplecnn_raw()
