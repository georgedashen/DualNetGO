import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dims):
        super(Autoencoder, self).__init__()
        
        # Define the encoder layers
        encoder_layers = []
        for i in range(len(encoding_dims)):
            if i == len(encoding_dims) // 2:
                encoder_layers.append(nn.Linear(input_dim, encoding_dims[i]))
                encoder_layers.append(nn.Sigmoid())
            else:
                encoder_layers.append(nn.Linear(encoding_dims[i-1], encoding_dims[i]))
                encoder_layers.append(nn.Sigmoid())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Define the decoder layer
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dims[-1], input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class MultimodalAutoencoder(nn.Module):
    def __init__(self, input_dims, encoding_dims):
        super(MultimodalAutoencoder, self).__init__()

        # Define the input layers
        self.input_layers = nn.ModuleList([nn.Linear(dim, int(encoding_dims[0] / len(input_dims))) for dim in input_dims])

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(int(encoding_dims[0] / len(input_dims)), int(encoding_dims[0] / len(input_dims)), nn.Sigmoid())])

        # Define the middle layers
        for i in range(1, len(encoding_dims) - 1):
            if i == len(encoding_dims) // 2:
                self.hidden_layers.append(nn.Linear(encoding_dims[i - 1], encoding_dims[i], nn.Sigmoid()))
            else:
                self.hidden_layers.append(nn.Linear(encoding_dims[i - 1], encoding_dims[i], nn.Sigmoid()))

        # Define the reconstruction layer
        self.reconstruction_layer = nn.Linear(encoding_dims[0], encoding_dims[0], nn.Sigmoid()) if len(encoding_dims) != 1 else None

        # Define the output layers
        self.output_layers = nn.ModuleList([nn.Linear(int(encoding_dims[-1] / len(input_dims)), dim, nn.Sigmoid()) for dim in input_dims])

    def forward(self, x):
        encoded = [F.silu(layer(x_i)) for layer, x_i in zip(self.input_layers, x)]
        hidden = [layer(encoded_i) for layer, encoded_i in zip(self.hidden_layers, encoded)]
        middle = self.reconstruction_layer(hidden[-1]) if self.reconstruction_layer is not None else None
        decoded = [layer(hidden_i) for layer, hidden_i in zip(self.output_layers, hidden)]
        return decoded
    
class build_MDA_encoder(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.input_layers = nn.ModuleList([nn.Linear(input_dim, 2500) for i in range(7)])
        self.hidden_1 = nn.ModuleList([nn.Linear(2500*7,9000), nn.Linear(9000,1200)])
        self.hidden_2 = nn.ModuleList([nn.Linear(1200,9000), nn.Linear(9000,2500*7)])
        self.output_layers = nn.ModuleList([nn.Linear(2500,input_dim) for i in range(7)])
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout)

        self.norm1 = nn.ModuleList([nn.LayerNorm(2500) for i in range(7)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(9000), nn.LayerNorm(1200)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(9000), nn.LayerNorm(2500*7)])

    def forward(self, src):
        outputs = []
        for i, layer in enumerate(self.input_layers):
            output = layer(src[i])
            output = self.norm1[i](output)
            output = self.activation(output)
            output = self.dropout(output)
            outputs.append(output)
        outputs = torch.cat(outputs,1)

        for i, layer in enumerate(self.hidden_1):
            outputs = layer(outputs)
            outputs = self.norm2[i](outputs)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)

        hs = outputs.clone()

        for i, layer in enumerate(self.hidden_2):
            outputs = layer(outputs)
            outputs = self.norm3[i](outputs)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)
        
        rec = []
        for i, layer in enumerate(self.output_layers):
            output = layer(outputs[:,i*2500:(i+1)*2500])
            rec.append(output)

        return rec, hs
        
