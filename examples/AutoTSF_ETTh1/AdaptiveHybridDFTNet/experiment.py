import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import traceback
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.fft import rfft, irfft

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series with boundary adjustment
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Enhanced series decomposition block with adaptive frequency selection
    """
    def __init__(self, kernel_size, freq_range=5, filter_strength=0.5, top_k=3):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.freq_range = freq_range
        self.filter_strength = filter_strength  # Controls how much filtering to apply
        self.top_k = top_k  # Number of top frequencies to enhance
        
    def _enhance_seasonal(self, seasonal):
        """Apply advanced frequency enhancement to seasonal component"""
        # Convert to frequency domain
        seasonal_fft = rfft(seasonal.permute(0, 2, 1), dim=2)
        power = torch.abs(seasonal_fft)**2
        
        # Find dominant frequencies (average across batch and channels)
        avg_power = torch.mean(power, dim=(0, 1))
        
        # Get top-k frequencies
        if len(avg_power) > self.top_k:
            # Find indices of top-k frequencies
            _, top_indices = torch.topk(avg_power, self.top_k)
            
            # Create a mask that emphasizes top-k frequencies and their neighbors
            mask = torch.ones_like(seasonal_fft) * (1 - self.filter_strength)
            
            # Enhance each top frequency and its neighbors
            for idx in top_indices:
                start_idx = max(0, idx - self.freq_range)
                end_idx = min(len(avg_power), idx + self.freq_range + 1)
                
                # Apply smoother enhancement with distance-based weighting
                for i in range(start_idx, end_idx):
                    # Calculate distance-based weight (closer = stronger enhancement)
                    distance = abs(i - idx)
                    weight = 1.0 - (distance / (self.freq_range + 1))
                    
                    # Apply weighted enhancement
                    mask[:, :, i] += weight * self.filter_strength
            
            # Apply mask and convert back to time domain
            filtered_fft = seasonal_fft * mask
            enhanced_seasonal = irfft(filtered_fft, dim=2, n=seasonal.size(1))
            return enhanced_seasonal.permute(0, 2, 1)
        
        # Fallback to simpler enhancement for small frequency ranges
        total_power = torch.sum(avg_power)
        if total_power > 0:
            freq_weights = avg_power / total_power
            # Smoother weight distribution
            freq_weights = freq_weights ** 0.3  # Less aggressive exponent
            
            # Apply weighted mask
            mask = torch.ones_like(seasonal_fft) * (1 - self.filter_strength)
            for i in range(len(freq_weights)):
                mask[:, :, i] += freq_weights[i] * self.filter_strength
                
            # Apply mask and convert back to time domain
            filtered_fft = seasonal_fft * mask
            enhanced_seasonal = irfft(filtered_fft, dim=2, n=seasonal.size(1))
            return enhanced_seasonal.permute(0, 2, 1)
        
        return seasonal  # Fallback to original if no power detected

    def forward(self, x):
        # Extract trend using moving average
        moving_mean = self.moving_avg(x)
        
        # Extract seasonal component (residual)
        seasonal = x - moving_mean
        
        # Apply advanced frequency enhancement
        enhanced_seasonal = self._enhance_seasonal(seasonal)
        
        # Blend original and enhanced seasonal with more weight on original
        # More conservative blending to maintain baseline performance
        final_seasonal = seasonal * 0.8 + enhanced_seasonal * 0.2
        
        return final_seasonal, moving_mean

# No replacement needed - we'll use a different approach


class SimpleTrendAttention(nn.Module):
    """
    Simple attention mechanism for trend component
    """
    def __init__(self, seq_len):
        super(SimpleTrendAttention, self).__init__()
        # Simple learnable attention weights
        self.attention = nn.Parameter(torch.ones(seq_len) / seq_len)
        
    def forward(self, x):
        # x: [Batch, seq_len, channels]
        # Apply attention weights along sequence dimension
        weights = F.softmax(self.attention, dim=0)
        # Reshape for broadcasting
        weights = weights.view(1, -1, 1)
        # Apply attention
        return x * weights


class AdaptiveHybridDFTNet(nn.Module):
    """
    Refined AdaptiveHybridDFTNet with balanced components
    """
    def __init__(self, configs):
        super(AdaptiveHybridDFTNet, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        
        # Dynamic kernel size selection based on sequence length
        kernel_size = min(25, max(5, self.seq_len // 8))
        kernel_size = configs.moving_avg if hasattr(configs, 'moving_avg') else kernel_size
        
        # Frequency range and filter strength
        freq_range = configs.freq_range if hasattr(configs, 'freq_range') else 5
        filter_strength = configs.filter_strength if hasattr(configs, 'filter_strength') else 0.2  # Reduced strength
        top_k = configs.top_k if hasattr(configs, 'top_k') else 3
        
        # Enhanced decomposition
        self.decomposition = series_decomp(kernel_size, freq_range, filter_strength, top_k)
        
        # Simple attention for trend
        self.trend_attention = SimpleTrendAttention(self.seq_len)
        
        # Linear projection layers (similar to baseline)
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
        # Learnable weights for combining seasonal and trend outputs
        self.seasonal_weight = nn.Parameter(torch.tensor(0.5))
        self.trend_weight = nn.Parameter(torch.tensor(0.5))
            
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        
        # Decompose with enhanced frequency selection
        seasonal, trend = self.decomposition(x)
        
        # Apply simple attention to trend
        trend = self.trend_attention(trend)
        
        # Convert to [Batch, Channel, Length] for linear projection
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        
        # Apply linear projection
        if self.individual:
            seasonal_output = torch.zeros([seasonal.size(0), self.pred_len, self.channels], 
                                         dtype=seasonal.dtype).to(seasonal.device)
            trend_output = torch.zeros([trend.size(0), self.pred_len, self.channels], 
                                      dtype=trend.dtype).to(trend.device)
            
            for i in range(self.channels):
                seasonal_output[:, :, i] = self.Linear_Seasonal[i](seasonal[:, i, :])
                trend_output[:, :, i] = self.Linear_Trend[i](trend[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal)
            trend_output = self.Linear_Trend(trend)
            
            # Convert back to [Batch, Length, Channel]
            seasonal_output = seasonal_output.permute(0, 2, 1)
            trend_output = trend_output.permute(0, 2, 1)
        
        # Normalize weights to sum to 1
        total_weight = torch.abs(self.seasonal_weight) + torch.abs(self.trend_weight)
        seasonal_weight_norm = torch.abs(self.seasonal_weight) / total_weight
        trend_weight_norm = torch.abs(self.trend_weight) / total_weight
        
        # Combine outputs with learnable weights
        x = seasonal_output * seasonal_weight_norm + trend_output * trend_weight_norm
        
        return x  # [Batch, Output length, Channel]


# For backward compatibility
class Model(AdaptiveHybridDFTNet):
    """
    Wrapper class for backward compatibility
    """
    def __init__(self, configs):
        super(Model, self).__init__(configs)


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument("--out_dir", type=str, default="run_0")
    # basic config
    
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average for trend extraction')
    parser.add_argument('--freq_range', type=int, default=5, help='frequency range for adaptive DFT selection')
    parser.add_argument('--filter_strength', type=float, default=0.2, help='strength of frequency filtering (0-1)')
    parser.add_argument('--top_k', type=int, default=3, help='number of top frequencies to enhance')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    try:
        log_dir = os.path.join(args.out_dir, 'logs')
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.dvices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        print('Args in experiment:')
        print(args)
        mse,mae = [], []
        pred_lens = [96, 192, 336, 720] if args.data_path != 'illness.csv' else [24, 36, 48, 60]
        for pred_len in pred_lens:
            args.pred_len = pred_len
            model = Model(args)
            Exp = Exp_Main
            setting = '{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des)

            exp = Exp(args,model)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting,writer)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            single_mae, single_mse = exp.test(setting)
            print('mse:{}, mae:{}'.format(single_mse, single_mae))
            mae.append(single_mae)
            mse.append(single_mse)
            torch.cuda.empty_cache()
        mean_mae = sum(mae) / len(mae)
        mean_mse = sum(mse) / len(mse)
        final_infos = {
            args.data :{
                "means":{
                    "mae": mean_mae,
                    "mse": mean_mse,
                }
            }
        }
        pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        # with open(os.path.join(args.out_dir, f"final_info_{args.data}.json"), "w") as f:
        with open(os.path.join(args.out_dir, f"final_info.json"), "w") as f:
            json.dump(final_infos, f) 
    
    except Exception as e:
        print("Original error in subprocess:", flush=True)
        traceback.print_exc(file=open(os.path.join(args.out_dir, "traceback.log"), "w"))
        raise
