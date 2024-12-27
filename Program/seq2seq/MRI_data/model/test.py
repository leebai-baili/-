import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.hidden_channels = hidden_channels
        
    def forward(self, x, hidden):
        h, c = hidden
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class MultiChannelConvLSTMEncoder(nn.Module):
    def __init__(self, input_channels_main, input_channels_blood, hidden_channels, kernel_size, num_layers):
        super(MultiChannelConvLSTMEncoder, self).__init__()
        
        # 主数据编码器
        self.main_encoder_layers = nn.ModuleList([
            ConvLSTMCell(input_channels_main if i == 0 else hidden_channels, hidden_channels, kernel_size)
            for i in range(num_layers)
        ])
        
        # 血流信息编码器
        self.blood_encoder_layers = nn.ModuleList([
            ConvLSTMCell(input_channels_blood if i == 0 else hidden_channels, hidden_channels, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x_main, x_blood):
        # 初始化隐藏状态
        b, t, _, h, w = x_main.size()
        hidden_main = [(torch.zeros(b, layer.hidden_channels, h, w, device=x_main.device),
                        torch.zeros(b, layer.hidden_channels, h, w, device=x_main.device))
                       for layer in self.main_encoder_layers]
        
        hidden_blood = [(torch.zeros(b, layer.hidden_channels, h, w, device=x_blood.device),
                         torch.zeros(b, layer.hidden_channels, h, w, device=x_blood.device))
                        for layer in self.blood_encoder_layers]

        outputs_main = []
        outputs_blood = []

        # 对主数据编码
        for time_step in range(t):
            input_t_main = x_main[:, time_step]
            for i, layer in enumerate(self.main_encoder_layers):
                hidden_main[i] = layer(input_t_main, hidden_main[i])
                input_t_main = hidden_main[i][0]
            outputs_main.append(input_t_main)
        
        # 对血流信息编码
        for time_step in range(t):
            input_t_blood = x_blood[:, time_step]
            for i, layer in enumerate(self.blood_encoder_layers):
                hidden_blood[i] = layer(input_t_blood, hidden_blood[i])
                input_t_blood = hidden_blood[i][0]
            outputs_blood.append(input_t_blood)
        
        # 将每个时间步的输出堆叠
        output_main = torch.stack(outputs_main, dim=1)
        output_blood = torch.stack(outputs_blood, dim=1)

        return output_main, output_blood  # 返回主数据和血流信息的编码表示

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        
    def forward(self, main_features, blood_features):
        combined = torch.cat((main_features, blood_features), dim=2)
        energy = torch.tanh(self.attn(combined))
        attention_weights = torch.softmax(energy @ self.v, dim=1)
        fused_context = (attention_weights.unsqueeze(2) * main_features).sum(dim=1)
        return fused_context

class ConditionalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConditionalDecoder, self).__init__()
        self.gru = nn.GRU(input_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)  # 输出层

    def forward(self, x, hidden, condition):
        # 将条件信息与输入拼接
        x_combined = torch.cat([x, condition], dim=2)
        output, hidden = self.gru(x_combined, hidden)
        output = self.fc(output)
        return output, hidden

# 综合使用多通道编码器、注意力融合和条件解码
class Seq2SeqWithAttentionAndCondition(nn.Module):
    def __init__(self, input_channels_main, input_channels_blood, hidden_channels, kernel_size, num_layers, output_dim):
        super(Seq2SeqWithAttentionAndCondition, self).__init__()
        self.encoder = MultiChannelConvLSTMEncoder(input_channels_main, input_channels_blood, hidden_channels, kernel_size, num_layers)
        self.attention = AttentionFusion(hidden_channels)
        self.decoder = ConditionalDecoder(input_dim=output_dim, hidden_dim=hidden_channels)

    def forward(self, x_main, x_blood, decoder_input):
        # 编码器阶段
        encoded_main, encoded_blood = self.encoder(x_main, x_blood)
        
        # 注意力融合阶段
        condition = self.attention(encoded_main, encoded_blood)
        
        # 条件解码阶段
        hidden = None  # 初始化解码器的隐状态
        output, _ = self.decoder(decoder_input, hidden, condition.unsqueeze(1).repeat(1, decoder_input.size(1), 1))
        return output
