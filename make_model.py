import torch
import torchcde

def make_model(config):
    if config.model.model_type == 'VAE':
        model = VAE(config)
    elif config.model.model_type == 'AE':
        model = AE(config)
    elif config.model.model_type in ['AE_AE', 'VAE_AE']:
        model = CTA(config)

    return model


def augment_time(data, feature_num):
    if data.shape[-1] == feature_num * 4: # if coeffs
        interpolated_path = torchcde.CubicSpline(data)
        data = interpolated_path.evaluate(interpolated_path.grid_points)
    batch_size, seq_len, _ = data.shape

    time_step = (torch.linspace(0, seq_len, seq_len) / seq_len).repeat(batch_size, 1).unsqueeze(-1).to(data.device)
    data_w_time = torch.cat([time_step, data], dim=-1)
    coeffs_w_time = torchcde.natural_cubic_coeffs(data_w_time)

    return coeffs_w_time



class Encoder(torch.nn.Module):
    def __init__(self, config, input_channels, hidden_channels, hidden_hidden_channels, num_layers):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels 
        self.latent_channels = hidden_channels // 2 # hidden = latent * 2
        self.hidden_hidden_channels = hidden_hidden_channels

        cde_func = torch.nn.ModuleList([torch.nn.Linear(hidden_channels, hidden_hidden_channels), torch.nn.SiLU()])
        for _ in range(num_layers):
            cde_func.append(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels))
            cde_func.append(torch.nn.SiLU())

        self.func = torch.nn.Sequential(
            *cde_func,
        )
        self.mu = torch.nn.Linear(hidden_hidden_channels, self.latent_channels * input_channels)
        self.sigma = torch.nn.Linear(hidden_hidden_channels, self.latent_channels * input_channels)

    def forward(self, t, z):
        enc = self.func(z) 
        z = torch.cat([self.mu(enc), self.sigma(enc)], dim=1)
        z = z.tanh()
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        
        return z



class CDEfunc(torch.nn.Module):
    def __init__(self, config, input_channels, hidden_channels, hidden_hidden_channels, num_layers):
        super(CDEfunc, self).__init__()
        self.config = config
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        cde_func = torch.nn.ModuleList([torch.nn.Linear(hidden_channels, hidden_hidden_channels), torch.nn.SiLU()])

        for _ in range(num_layers):
            cde_func.append(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels))
            cde_func.append(torch.nn.SiLU())

        self.func = torch.nn.Sequential(
            
            *cde_func,
            torch.nn.Linear(hidden_hidden_channels, hidden_channels * input_channels)
        )

    def forward(self, t, z):
        z = self.func(z) 
        z = z.tanh()
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)

        return z


class AE(torch.nn.Module):
    def __init__(self, config, hidden_channels, latent_channels, num_layers, step_size):
        super(AE, self).__init__()
        self.config = config
        self.seq_len = config.data.seq_len
        self.feature_num = config.data.feature_num

        self.input_channels = config.data.feature_num + 1 # time augment
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.output_channels =  config.data.feature_num

        self.num_layers = num_layers

        # initial z0
        self.initial_encoder = torch.nn.Linear(self.input_channels, self.latent_channels)
        self.initial_decoder = torch.nn.Linear(self.input_channels, self.output_channels)

        # cde function 
        self.func_encoder = CDEfunc(config, self.input_channels, self.latent_channels, self.hidden_channels, self.num_layers)
        self.func_decoder = CDEfunc(config, self.latent_channels+1, self.output_channels, self.hidden_channels, self.num_layers)
        
        # readout
        self.readout_e_hat = torch.nn.Sequential(
            torch.nn.Linear(self.output_channels, self.output_channels*4),
            torch.nn.ELU(),
            torch.nn.Linear(self.output_channels*4, self.output_channels)  
        )
        
        # cdeint argument
        self.step_size = step_size
        self.atol = config.training.atol
        self.rtol = config.training.rtol
        self.solver = config.training.solver
        self.options = {"step_size":self.step_size}


    def get_initial_value(self, X0):
        z_encoder = self.initial_encoder(X0)
        z_decoder = self.initial_decoder(X0)
        return (z_encoder, z_decoder)


    def forward(self, coeffs, repalce_value=None, is_test=None):
        coeffs = augment_time(coeffs, self.feature_num)

        X = torchcde.CubicSpline(coeffs)
        X0 = X.evaluate(X.interval[0])  
        z0 = self.get_initial_value(X0)        

        adjoint_params=tuple(self.func_encoder.parameters()) + tuple(self.func_decoder.parameters())
        
        z_T = torchcde.cdeint_CVA(X=X, z0=z0,
                                   func_encoder=self.func_encoder, func_decoder=self.func_decoder,
                                   config=self.config,
                                   t=X.grid_points,
                                   submodel='AE',
                                   atol=self.atol,rtol=self.rtol,
                                   method=self.solver, options=self.options, adjoint_params=adjoint_params)

        encode, decode = z_T

        pred_y = self.readout_e_hat(decode) # ..., time_step, output_channels

        if repalce_value is not None:
            pred_y = repalce_value + pred_y
            
        return (), pred_y, encode


class VAE(torch.nn.Module):
    def __init__(self, config, hidden_channels, latent_channels, num_layers, step_size):
        super(VAE, self).__init__()
        self.config = config
        self.seq_len = config.data.seq_len
        self.feature_num = config.data.feature_num

        self.input_channels = config.data.feature_num + 1 # time augment
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.output_channels =  config.data.feature_num

        self.num_layers = num_layers

        # initial z0
        self.initial_encoder = torch.nn.Linear(self.input_channels, self.latent_channels*2)
        self.initial_decoder = torch.nn.Linear(self.input_channels, self.output_channels)

        # cde function 
        self.func_encoder = Encoder(config, self.input_channels, self.latent_channels*2, self.hidden_channels, self.num_layers)
        self.func_decoder = CDEfunc(config, self.latent_channels+1, self.output_channels, self.hidden_channels,self.num_layers)

        self.readout_e_hat = torch.nn.Sequential(
            torch.nn.Linear(self.output_channels, self.output_channels*4),
            torch.nn.ELU(),
            torch.nn.Linear(self.output_channels*4, self.output_channels)  
        )

        # cdeint argument
        self.atol = config.training.atol
        self.rtol = config.training.rtol
        self.solver = config.training.solver
        self.step_size = step_size

        self.options = {"step_size":self.step_size}


    def get_initial_value(self, X0):
        z_encoder = self.initial_encoder(X0)
        z_decoder = self.initial_decoder(X0)
        z_kld = torch.zeros_like(X0[:,0])
        return (z_encoder, z_decoder, z_kld)

    def forward(self, coeffs, repalce_value=None, is_test=None):
        coeffs = augment_time(coeffs, self.feature_num)
   
        X = torchcde.CubicSpline(coeffs)
        X0 = X.evaluate(X.interval[0])  
        z0 = self.get_initial_value(X0)        

        adjoint_params=tuple(self.func_encoder.parameters()) + tuple(self.func_decoder.parameters())
        z_T = torchcde.cdeint_CVA(X=X, z0=z0,
                                   func_encoder=self.func_encoder, func_decoder=self.func_decoder,
                                   config=self.config,
                                   t=X.grid_points,
                                   submodel='VAE_contKLD',
                                   is_test=is_test,
                                   atol=self.atol,rtol=self.rtol,
                                   method=self.solver, options=self.options, adjoint_params=adjoint_params)
        
        encode, decode, kld = z_T

        kld = kld[:,-1]
 
        pred_y = self.readout_e_hat(decode) 

        if repalce_value is not None:
            pred_y = repalce_value + pred_y

        return (), pred_y, kld 


class CTA(torch.nn.Module):
    def __init__(self, config):
        super(CTA, self).__init__()
        self.config = config
        self.block_types = config.model.model_type.split('_')

        if config.model.model_type == 'AE_AE':
            self.FirstBlock = AE(config, config.model.first_hidden_channels, config.model.first_latent_channels, config.model.first_num_layers, config.model.first_step_size)
            self.SecondBlock = AE(config, config.model.second_hidden_channels, config.model.second_latent_channels, config.model.second_num_layers, config.model.second_step_size)
        elif config.model.model_type == 'VAE_AE':
            self.FirstBlock = VAE(config, config.model.first_hidden_channels, config.model.first_latent_channels, config.model.first_num_layers, config.model.first_step_size)
            self.SecondBlock = AE(config, config.model.second_hidden_channels, config.model.second_latent_channels, config.model.second_num_layers, config.model.second_step_size)
        # self.FirstBlock, 
        feature_num = config.data.feature_num
        second_latent_channels = config.model.second_latent_channels

        self.weight_combine = torch.nn.Linear(feature_num + second_latent_channels, feature_num)
        self.sigmoid = torch.nn.Sigmoid()

        
    def forward(self, coeffs, origin_X, M, is_test=None):
        # First block
        # if VAE, side_info is KLD 
        # if AE, side_info is encode
        _, X_hat, side_info = self.FirstBlock.forward(coeffs, is_test=is_test)

        if self.block_types[0] == 'AE':
            # side info will be used as KLD
            side_info = torch.zeros_like(side_info)

        # replace first prediction
        X_check = M * origin_X + (1 - M) * X_hat
        
        # Second block
        _, X_hat_prime, d_prime = self.SecondBlock.forward(X_check, repalce_value=X_check)

        combining_weights = self.sigmoid(self.weight_combine(torch.cat([d_prime, M], dim=2)))  # namely term alpha

        X_tilde = combining_weights * X_hat + (1-combining_weights) * X_hat_prime


        return (X_hat, X_hat_prime), X_tilde, side_info
