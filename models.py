import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from resnet import Resnet1D
from vector_quantize_pytorch import VectorQuantize, FSQ
import whisper
from transformers import AutoModelForCausalLM
import types
import math

from peft import LoraConfig, get_peft_model, TaskType

########################################################################################################
######                                      Tokenizer Modules                                     ###### 
########################################################################################################         
class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    
class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    

class FaceMotionTokenizer(nn.Module):

    def __init__(self, input_dim=181, down_t=3, stride_t=2, quantizer='vq', embed=256, codebook_size=1024, levels=[8, 5, 5, 5]):
        super().__init__()

        self.enc = Encoder(input_emb_width=input_dim, output_emb_width=embed, down_t=down_t, stride_t=stride_t)
        self.dec = Decoder(input_emb_width=input_dim, output_emb_width=embed, down_t=down_t, stride_t=stride_t)

        self.quantizer = quantizer

        if quantizer=='vq':
            self.quantize = VectorQuantize(
                dim=embed,
                codebook_size=codebook_size,     # codebook size
                decay=0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight=1.   # the weight on the commitment loss
            )
        elif quantizer=='fsq':
            self.quantize = FSQ(
                levels=levels,
                dim=embed
            )
        else:
            raise Exception(f"quantizer has to be either vq or fsq - {quantizer} given!")
        
    def encode(self, x):
        
        x = torch.permute(x, (0, 2, 1))
        x = self.enc(x)
        x = torch.permute(x, (0, 2, 1))

        return x
    
    def tokenize(self, x):
        x = self.encode(x)

        if self.quantizer=='vq':
            quantized, indices, commit_loss = self.quantize(x)
        elif self.quantizer=='fsq':
            quantized, indices = self.quantize(x)
            commit_loss = None

        return quantized, indices, commit_loss
    
    def decode(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.dec(x)
        x = torch.permute(x, (0, 2, 1))

        return x
    
    def indices2code(self, indices):
        if self.quantizer=='vq':
            codes = self.quantize.get_codes_from_indices(indices)
        elif self.quantizer=='fsq':
            codes = self.quantize.indices_to_codes(indices)

        return codes
    
    def forward(self, x):

        quantized, indices, commit_loss = self.tokenize(x)

        recon = self.decode(quantized)

        return recon, commit_loss

class FaceMotionTokenizerV2(nn.Module):

    def __init__(self, input_dim=181, down_t=3, stride_t=2, quantizer='vq', embed=256, codebook_size=1024, levels=[8, 5, 5, 5]):
        super().__init__()

        self.enc = Encoder(input_emb_width=input_dim, output_emb_width=embed, down_t=down_t, stride_t=stride_t)
        self.dec_exp = Decoder(input_emb_width=171, output_emb_width=embed, down_t=down_t, stride_t=stride_t)
        self.dec_pose = Decoder(input_emb_width=6, output_emb_width=embed, down_t=down_t, stride_t=stride_t)
        self.dec_eye = Decoder(input_emb_width=4, output_emb_width=embed, down_t=down_t, stride_t=stride_t)

        self.quantizer = quantizer

        if quantizer=='vq':
            self.quantize = VectorQuantize(
                dim=embed,
                codebook_size=codebook_size,     # codebook size
                decay=0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight=1.   # the weight on the commitment loss
            )
        elif quantizer=='fsq':
            self.quantize = FSQ(
                levels=levels,
                dim=embed
            )
        else:
            raise Exception(f"quantizer has to be either vq or fsq - {quantizer} given!")
        
    def encode(self, x):
        
        x = torch.permute(x, (0, 2, 1))
        x = self.enc(x)
        x = torch.permute(x, (0, 2, 1))

        return x
    
    def tokenize(self, x):
        x = self.encode(x)

        if self.quantizer=='vq':
            quantized, indices, commit_loss = self.quantize(x)
        elif self.quantizer=='fsq':
            quantized, indices = self.quantize(x)
            commit_loss = None

        return quantized, indices, commit_loss
    
    def decode(self, x):
        x = torch.permute(x, (0, 2, 1))

        exp = self.dec_exp(x)
        exp = torch.permute(exp, (0, 2, 1))

        pose = self.dec_pose(x)
        pose = torch.permute(pose, (0, 2, 1))

        eye = self.dec_eye(x)
        eye = torch.permute(eye, (0, 2, 1))

        motion = torch.cat([exp, pose, eye], dim=-1)

        return motion
    
    def indices2code(self, indices):
        if self.quantizer=='vq':
            codes = self.quantize.get_codes_from_indices(indices)
        elif self.quantizer=='fsq':
            codes = self.quantize.indices_to_codes(indices)

        return codes
    
    def forward(self, x):

        quantized, indices, commit_loss = self.tokenize(x)

        recon = self.decode(quantized)

        return recon, commit_loss
    

class ReconLoss(nn.Module):
    def __init__(self, exp_weight, pose_weight, eye_weight):
        super().__init__()
        self.exp_weight = exp_weight
        self.pose_weight = pose_weight
        self.eye_weight = eye_weight

    def forward(self, pred, gt):

        exp_loss = F.mse_loss(pred[:, :, :171], gt[:, :, :171])
        pose_loss = F.mse_loss(pred[:, :, 171:-4], gt[:, :, 171:-4])
        eye_loss = F.mse_loss(pred[:, :, -4:], gt[:, :, -4:])

        loss = self.exp_weight * exp_loss + self.pose_weight * pose_loss + self.eye_weight * eye_loss

        return loss
    
class VelLoss(nn.Module):

    def __init__(self, exp_weight, pose_weight, eye_weight):
        super().__init__()
        self.exp_weight = exp_weight
        self.pose_weight = pose_weight
        self.eye_weight = eye_weight

    def forward(self, pred, gt):
        vel_pred = torch.cat((
            torch.zeros_like(pred[:,:1,:]),
            pred[:,1:,:]-pred[:,:-1,:]
        ), dim=1)

        vel_gt = torch.cat((
            torch.zeros_like(gt[:,:1,:]),
            gt[:,1:,:]-gt[:,:-1,:]
        ), dim=1)

        exp_loss = F.mse_loss(vel_pred[:, :, :171], vel_gt[:, :, :171])
        pose_loss = F.mse_loss(vel_pred[:, :, 171:-4], vel_gt[:, :, 171:-4])
        eye_loss = F.mse_loss(vel_pred[:, :, -4:], vel_gt[:, :, -4:])

        loss = self.exp_weight * exp_loss + self.pose_weight * pose_loss + self.eye_weight * eye_loss

        return loss


########################################################################################################
######                                      Predictor Modules                                     ###### 
######################################################################################################## 

class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)

        self.ln = nn.LayerNorm(config.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        x = self.ln(x)
        return x

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls):
        
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        # encoder = WhisperModel.from_pretrained('openai/whisper-large-v3',torch_dtype=torch.bfloat16).encoder
        encoder = whisper.load_model(name='large', device='cpu').encoder

        d_model = encoder.conv2.out_channels   # 1280 for large-v3
        n_ctx = 1500                           # max sequence length
        encoder.positional_embedding = nn.Parameter(
            torch.empty(n_ctx, d_model)
        )
        nn.init.normal_(encoder.positional_embedding, std=0.02)

        
        encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        # encoder = whisper.load_model(name='large', device='cpu').encoder
        # encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)

        return encoder
    
class FPSResampler(nn.Module):
    def __init__(self, feats, in_fps, out_fps, kernel_size=3):
        super().__init__()
        self.in_fps = in_fps
        self.out_fps = out_fps
        
        if out_fps < in_fps:  # downsample
            stride = int(round(in_fps / out_fps))
            self.conv = nn.Conv1d(
                in_channels=feats,
                out_channels=feats,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            )
            self.is_transpose = False
        else:  # upsample
            stride = int(round(out_fps / in_fps))
            self.conv = nn.ConvTranspose1d(
                in_channels=feats,
                out_channels=feats,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                output_padding=stride - 1  # adjust length
            )
            self.is_transpose = True

    def forward(self, x):
        # x: (batch, seq1, feats)
        x = x.transpose(1, 2)   # -> (batch, feats, seq1)
        x = self.conv(x)        # resample
        x = x.transpose(1, 2)   # -> (batch, seq2, feats)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return x
    
class FinalModel(nn.Module):

    def __init__(self, config, freeze_llm=True):
        super().__init__()

        self.encoder = WhisperWrappedEncoder().load()
        for params in self.encoder.parameters():
            params.requires_grad = False

        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model).to(dtype=torch.bfloat16)

        if freeze_llm:
            for params in self.llm.parameters():
                params.requires_grad = False
        else:
            for params in self.llm.parameters():
                params.requires_grad = True
            
        self.projector = EncoderProjectorConcat(config).to(dtype=torch.bfloat16)

        self.downsampler = FPSResampler(feats=config.llm_dim, in_fps=config.in_fps, out_fps=config.out_fps, kernel_size=config.kernel_size).to(dtype=torch.bfloat16)

        self.feat_projection = nn.Linear(config.llm_dim, config.num_vq).to(dtype=torch.bfloat16)

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.num_vq, nhead=config.nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers).to(dtype=torch.bfloat16)


        self.pe = PositionalEncoding(d_model=config.num_vq).to(dtype=torch.bfloat16)

    def get_closest_embeddings(self, x, top_k: int = 1):
        """
        Given input vectors x (batch, seq, dim), find the closest embeddings
        from the embedding table.

        Args:
            x: torch.Tensor of shape (batch, seq, dim)
            embedding: nn.Embedding object
            top_k: how many nearest embeddings to return

        Returns:
            closest_embeds: torch.Tensor of shape (batch, seq, top_k, dim)
            indices: torch.LongTensor of shape (batch, seq, top_k)
        """
        # Get embedding weights
        emb_weights = self.llm.model.embed_tokens.weight  # (num_tokens, dim)

        # Normalize for cosine similarity
        x_norm = F.normalize(x, dim=-1)              # (batch, seq, dim)
        emb_norm = F.normalize(emb_weights, dim=-1)  # (num_tokens, dim)

        # Similarity: (batch, seq, num_tokens)
        similarities = torch.matmul(x_norm, emb_norm.T)

        # Top-k indices
        values, indices = similarities.topk(top_k, dim=-1)  # (batch, seq, k)

        # Gather closest embeddings
        closest_embeds = emb_weights[indices]  # (batch, seq, k, dim)
        closest_embeds = closest_embeds.squeeze(2)

        return closest_embeds, indices

    def forward(self, audio_mel, max_tokens=48):

        audio_encoded = self.encoder.extract_variable_length_features(audio_mel)
        audio_encoded = self.projector(audio_encoded)

        bs, seq = audio_encoded.shape[0], audio_encoded.shape[1]
        attention_mask = torch.ones((bs, seq), dtype=torch.long, device=audio_encoded.device)

        logits = self.llm(inputs_embeds=audio_encoded, output_hidden_states=True, attention_mask=attention_mask).hidden_states[-1]

        logits = self.downsampler(logits)

        logits = self.feat_projection(logits)

        logits = self.pe(logits)

        memory = torch.zeros(bs, logits.shape[1], logits.shape[2]).to(device=logits.device, dtype=logits.dtype)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(logits.shape[1]).to(logits.device)
        logits = self.transformer_decoder(logits, memory=memory, tgt_mask=causal_mask)

        logits = logits[:, :max_tokens, :]

        return logits
    

class FinalModelEmbed(nn.Module):

    def __init__(self, config, freeze_llm=True):
        super().__init__()

        self.encoder = WhisperWrappedEncoder().load()
        for params in self.encoder.parameters():
            params.requires_grad = False

        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model).to(dtype=torch.bfloat16)

        if freeze_llm:
            for params in self.llm.parameters():
                params.requires_grad = False
        else:
            for params in self.llm.parameters():
                params.requires_grad = True
            
        self.projector = EncoderProjectorConcat(config).to(dtype=torch.bfloat16)

        self.downsampler = FPSResampler(feats=config.llm_dim, in_fps=config.in_fps, out_fps=config.out_fps, kernel_size=config.kernel_size).to(dtype=torch.bfloat16)

        self.feat_projection = nn.Linear(config.llm_dim, config.num_vq).to(dtype=torch.bfloat16)

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.num_vq, nhead=config.nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers).to(dtype=torch.bfloat16)


        self.pe = PositionalEncoding(d_model=config.num_vq).to(dtype=torch.bfloat16)

    def get_closest_embeddings(self, x, top_k: int = 1):
        """
        Given input vectors x (batch, seq, dim), find the closest embeddings
        from the embedding table.

        Args:
            x: torch.Tensor of shape (batch, seq, dim)
            embedding: nn.Embedding object
            top_k: how many nearest embeddings to return

        Returns:
            closest_embeds: torch.Tensor of shape (batch, seq, top_k, dim)
            indices: torch.LongTensor of shape (batch, seq, top_k)
        """
        # Get embedding weights
        emb_weights = self.llm.model.embed_tokens.weight  # (num_tokens, dim)

        # Normalize for cosine similarity
        x_norm = F.normalize(x, dim=-1)              # (batch, seq, dim)
        emb_norm = F.normalize(emb_weights, dim=-1)  # (num_tokens, dim)

        # Similarity: (batch, seq, num_tokens)
        similarities = torch.matmul(x_norm, emb_norm.T)

        # Top-k indices
        values, indices = similarities.topk(top_k, dim=-1)  # (batch, seq, k)

        # Gather closest embeddings
        closest_embeds = emb_weights[indices]  # (batch, seq, k, dim)
        closest_embeds = closest_embeds.squeeze(2)

        return closest_embeds, indices

    def forward(self, audio_mel, max_tokens=48):

        audio_encoded = self.encoder.extract_variable_length_features(audio_mel)
        audio_encoded = self.projector(audio_encoded)

        audio_embed, indices = self.get_closest_embeddings(audio_encoded)

        bs, seq = audio_encoded.shape[0], audio_encoded.shape[1]
        attention_mask = torch.ones((bs, seq), dtype=torch.long, device=audio_encoded.device)

        logits = self.llm(inputs_embeds=audio_embed, output_hidden_states=True, attention_mask=attention_mask).hidden_states[-1]

        logits = self.downsampler(logits)

        logits = self.feat_projection(logits)

        logits = self.pe(logits)

        memory = torch.zeros(bs, logits.shape[1], logits.shape[2]).to(device=logits.device, dtype=logits.dtype)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(logits.shape[1]).to(logits.device)
        logits = self.transformer_decoder(logits, memory=memory, tgt_mask=causal_mask)

        logits = logits[:, :max_tokens, :]

        commit_loss = F.mse_loss(audio_encoded, audio_embed.detach())

        return logits, commit_loss

class FinalModelEmbedProj(nn.Module):

    def __init__(self, config, freeze_llm=True, lora=False):
        super().__init__()

        self.encoder = WhisperWrappedEncoder().load()
        for params in self.encoder.parameters():
            params.requires_grad = False

        for block in self.encoder.blocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True

        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model).to(dtype=torch.bfloat16)

        if freeze_llm:
            for params in self.llm.parameters():
                params.requires_grad = False
        else:
            for params in self.llm.parameters():
                params.requires_grad = True

        self.lora = lora
        if self.lora:
            print('Using LoRA')
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=['q_proj', 'k_proj', 'gate_proj', 'up_proj', 'down_proj'],
            )

            self.llm = get_peft_model(self.llm, lora_config)

            
        self.projector = EncoderProjectorConcat(config).to(dtype=torch.bfloat16)

        self.downsampler = FPSResampler(feats=config.llm_dim, in_fps=config.in_fps, out_fps=config.out_fps, kernel_size=config.kernel_size).to(dtype=torch.bfloat16)

        self.feat_projection = nn.Linear(config.llm_dim, config.num_vq).to(dtype=torch.bfloat16)


    def get_closest_embeddings(self, x, top_k: int = 1):
        """
        Given input vectors x (batch, seq, dim), find the closest embeddings
        from the embedding table.

        Args:
            x: torch.Tensor of shape (batch, seq, dim)
            embedding: nn.Embedding object
            top_k: how many nearest embeddings to return

        Returns:
            closest_embeds: torch.Tensor of shape (batch, seq, top_k, dim)
            indices: torch.LongTensor of shape (batch, seq, top_k)
        """
        # Get embedding weights
        if self.lora:
            emb_weights = self.llm.base_model.model.model.embed_tokens.weight  # (num_tokens, dim)
        else:
            emb_weights = self.llm.model.embed_tokens.weight  # (num_tokens, dim)

        # Normalize for cosine similarity
        x_norm = F.normalize(x, dim=-1)              # (batch, seq, dim)
        emb_norm = F.normalize(emb_weights, dim=-1)  # (num_tokens, dim)

        # Similarity: (batch, seq, num_tokens)
        similarities = torch.matmul(x_norm, emb_norm.T)

        # Top-k indices
        values, indices = similarities.topk(top_k, dim=-1)  # (batch, seq, k)

        # Gather closest embeddings
        closest_embeds = emb_weights[indices]  # (batch, seq, k, dim)
        closest_embeds = closest_embeds.squeeze(2)

        return closest_embeds, indices

    def forward(self, audio_mel, max_tokens=48):

        audio_encoded = self.encoder.extract_variable_length_features(audio_mel)
        audio_encoded = self.projector(audio_encoded)

        audio_embed, indices = self.get_closest_embeddings(audio_encoded)

        bs, seq = audio_encoded.shape[0], audio_encoded.shape[1]
        attention_mask = torch.ones((bs, seq), dtype=torch.long, device=audio_encoded.device)

        logits = self.llm(inputs_embeds=audio_embed, output_hidden_states=True, attention_mask=attention_mask).hidden_states[-1]

        logits = self.downsampler(logits)

        logits = self.feat_projection(logits)

        logits = logits[:, :max_tokens, :]

        commit_loss = F.mse_loss(audio_encoded, audio_embed.detach())

        return logits, commit_loss