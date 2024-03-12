from typing import Tuple
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from squeezeformer.attention import MultiHeadedSelfAttentionModule
from squeezeformer.convolution import ConvModule, DepthwiseConv2dSubsampling, TimeReductionLayer
from squeezeformer.modules import FeedForwardModule, ResidualConnectionModule, recover_resolution

class SqueezeformerBlock(nn.Module):
    """
    SqueezeformerBlock is a simpler block structure similar to the standard Transformer block,
    where the MHA and convolution modules are each directly followed by a single feed forward module.

    Args:
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by squeezeformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = False,
    ):
        super(SqueezeformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1.0

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            nn.LayerNorm(encoder_dim),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
            ResidualConnectionModule(
                module=ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            nn.LayerNorm(encoder_dim),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)

class SqueezeformerEncoder(nn.Module):
    """
    Squeezeformer encoder first processes the input with a convolution subsampling layer and then
    with a number of squeezeformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector 
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        reduce_layer_index: int = 7,
        recover_layer_index: int = 15,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = False,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.reduce_layer_index = reduce_layer_index
        self.recover_layer_index = recover_layer_index
        self.conv_subsample = DepthwiseConv2dSubsampling(in_channels=1, out_channels=encoder_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            ###nn.Linear(encoder_dim,encoder_dim)原本的设计是配合conv_subsample的输出用的,可能可以用FeatureExtractor替代
            nn.Dropout(p=input_dropout_p),
        )
        self.time_reduction_layer = TimeReductionLayer()
        self.time_reduction_proj = nn.Linear((encoder_dim - 1) // 2, encoder_dim)
        self.time_recover_layer = nn.Linear(encoder_dim, encoder_dim)
        self.recover_tensor = None

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            if idx < reduce_layer_index:
                self.layers.append(
                    SqueezeformerBlock(
                        encoder_dim=encoder_dim,
                        num_attention_heads=num_attention_heads,
                        feed_forward_expansion_factor=feed_forward_expansion_factor,
                        conv_expansion_factor=conv_expansion_factor,
                        feed_forward_dropout_p=feed_forward_dropout_p,
                        attention_dropout_p=attention_dropout_p,
                        conv_dropout_p=conv_dropout_p,
                        conv_kernel_size=conv_kernel_size,
                        half_step_residual=half_step_residual,
                    )
                )
            elif reduce_layer_index <= idx < recover_layer_index:
                self.layers.append(
                    ResidualConnectionModule(
                        module=SqueezeformerBlock(
                            encoder_dim=encoder_dim,
                            num_attention_heads=num_attention_heads,
                            feed_forward_expansion_factor=feed_forward_expansion_factor,
                            conv_expansion_factor=conv_expansion_factor,
                            feed_forward_dropout_p=feed_forward_dropout_p,
                            attention_dropout_p=attention_dropout_p,
                            conv_dropout_p=conv_dropout_p,
                            conv_kernel_size=conv_kernel_size,
                            half_step_residual=half_step_residual,
                        )
                    )
                )
            else:
                self.layers.append(
                    SqueezeformerBlock(
                        encoder_dim=encoder_dim,
                        num_attention_heads=num_attention_heads,
                        feed_forward_expansion_factor=feed_forward_expansion_factor,
                        conv_expansion_factor=conv_expansion_factor,
                        feed_forward_dropout_p=feed_forward_dropout_p,
                        attention_dropout_p=attention_dropout_p,
                        conv_dropout_p=conv_dropout_p,
                        conv_kernel_size=conv_kernel_size,
                        half_step_residual=half_step_residual,
                    )
                )

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        return sum([p.numel for p in self.parameters()])

    def forward(self,inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:  #
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        
        #outputs, output_lengths = self.conv_subsample(inputs,input_lengths )
        outputs = inputs
        output_lengths = input_lengths
        #outputs = self.input_proj(outputs)

        for idx, layer in enumerate(self.layers):
            if idx == self.reduce_layer_index:
                self.recover_tensor = outputs
                outputs, output_lengths = self.time_reduction_layer(outputs, output_lengths)
                outputs = self.time_reduction_proj(outputs)

            if idx == self.recover_layer_index:
                outputs = recover_resolution(outputs)
                length = outputs.size(1)
                outputs = self.time_recover_layer(outputs)
                outputs += self.recover_tensor[:, :length, :]
                output_lengths *= 2

            outputs = layer(outputs)

        return outputs, output_lengths

class Squeezeformer_RNA(nn.Module):
    """
    Squeezeformer incorporates the Temporal U-Net structure, which reduces the cost of the
    multi-head attention modules on long sequences, and a simpler block structure of feed-forward module,
    followed up by multi-head attention or convolution modules,
    instead of the Macaron structure proposed in Conformer.

    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_encoder_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self,cfg,infer_mode=False,) -> None:
        super(Squeezeformer_RNA, self).__init__() 
        self.encoder = SqueezeformerEncoder(
            input_dim=cfg.encoder_config.input_dim,
            encoder_dim=cfg.encoder_config.encoder_dim,
            num_layers=cfg.encoder_config.num_encoder_layers,
            reduce_layer_index=cfg.encoder_config.reduce_layer_index,
            recover_layer_index=cfg.encoder_config.recover_layer_index,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor = cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor = cfg.encoder_config.conv_expansion_factor,
            input_dropout_p = cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p = cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p = cfg.encoder_config.attention_dropout_p,
            conv_dropout_p = cfg.encoder_config.conv_dropout_p,
            conv_kernel_size = cfg.encoder_config.conv_kernel_size,
            half_step_residual = cfg.encoder_config.half_step_residual,
        )
        
        self.token_embeddings = nn.Embedding(4,cfg.encoder_config.encoder_dim)
        #self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_d)
        self.loss= importlib.import_module(cfg.loss).loss
        self.fc = nn.Linear(cfg.encoder_config.encoder_dim, 2)
        self.out=torch.nn.Sigmoid()
        self.infer_mode=infer_mode

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        return self.encoder.count_parameters()

    def forward(self,x) -> Tuple[Tensor, Tensor]:  #inputs: Tensor, input_lengths: Tensor
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        if self.infer_mode:
            x2=x
            inputs = x2['inputs'] 
            seq=x2['seq'] #bpp output
            input_lengths = x2['input_lengths']
            inputs =self.token_embeddings(inputs)
            inputs=torch.matmul(seq,inputs)
            encoder_outputs, encoder_output_lengths = self.encoder(inputs,input_lengths)
            fc_outputs = self.fc(encoder_outputs)
            outputs={}
            outputs['fc_outputs']=fc_outputs
            return outputs
        else:
            x2=x[0]
            inputs = x2['inputs'] 
            seq=x2['seq']
            input_lengths = x2['input_lengths']
            inputs =self.token_embeddings(inputs)
            inputs=torch.matmul(seq,inputs)
            encoder_outputs, encoder_output_lengths = self.encoder(inputs,input_lengths)
            fc_outputs = self.fc(encoder_outputs)
            #outputs = F.log_softmax(outputs, dim=-1)
            #outputs = self.out(outputs)
            outputs={}
            outputs['loss']=self.loss(fc_outputs,x[1])
            outputs['fc_outputs']=fc_outputs
            return outputs #, encoder_output_lengths
    
class Squeezeformer_RNA_debug(nn.Module):
    """
    Squeezeformer incorporates the Temporal U-Net structure, which reduces the cost of the
    multi-head attention modules on long sequences, and a simpler block structure of feed-forward module,
    followed up by multi-head attention or convolution modules,
    instead of the Macaron structure proposed in Conformer.

    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_encoder_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self,cfg,infer_mode=False,) -> None:
        super(Squeezeformer_RNA, self).__init__() 
        self.encoder = SqueezeformerEncoder(
            input_dim=cfg.encoder_config.input_dim,
            encoder_dim=cfg.encoder_config.encoder_dim,
            num_layers=cfg.encoder_config.num_encoder_layers,
            reduce_layer_index=cfg.encoder_config.reduce_layer_index,
            recover_layer_index=cfg.encoder_config.recover_layer_index,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor = cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor = cfg.encoder_config.conv_expansion_factor,
            input_dropout_p = cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p = cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p = cfg.encoder_config.attention_dropout_p,
            conv_dropout_p = cfg.encoder_config.conv_dropout_p,
            conv_kernel_size = cfg.encoder_config.conv_kernel_size,
            half_step_residual = cfg.encoder_config.half_step_residual,
        )
        
        self.token_embeddings = nn.Embedding(4,cfg.encoder_config.encoder_dim)
        #self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_d)
        self.loss= importlib.import_module(cfg.loss).loss
        self.fc = nn.Linear(cfg.encoder_config.encoder_dim, 2)
        self.out=torch.nn.Sigmoid()
        self.infer_mode=infer_mode

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        return self.encoder.count_parameters()

    def forward(self,x) -> Tuple[Tensor, Tensor]:  #inputs: Tensor, input_lengths: Tensor
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        if self.infer_mode:
            x2=x
            inputs = x2['inputs'] 
            seq=x2['seq'] #bpp output
            input_lengths = x2['input_lengths']
            inputs =self.token_embeddings(inputs)
            #inputs=torch.matmul(seq,inputs)
            encoder_outputs, encoder_output_lengths = self.encoder(inputs,input_lengths)
            fc_outputs = self.fc(encoder_outputs)
            outputs={}
            outputs['fc_outputs']=fc_outputs
            return outputs
        else:
            x2=x[0]
            inputs = x2['inputs'] 
            seq=x2['seq']
            input_lengths = x2['input_lengths']
            inputs =self.token_embeddings(inputs)
            #inputs=torch.matmul(seq,inputs)
            encoder_outputs, encoder_output_lengths = self.encoder(inputs,input_lengths)
            fc_outputs = self.fc(encoder_outputs)
            #outputs = F.log_softmax(outputs, dim=-1)
            #outputs = self.out(outputs)
            outputs={}
            outputs['loss']=self.loss(fc_outputs,x[1])
            outputs['fc_outputs']=fc_outputs
            return outputs #, encoder_output_lengths