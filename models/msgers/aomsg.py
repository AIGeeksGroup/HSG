# decoder style AoMSG
# set the module as a separate file for cleaness
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models.loss import get_match_idx, get_association_sv
import numpy as np
import math
from models import lorentz as L
from models import distributed as dist
from models.loss import InfoNCELoss, MaskBCELoss, FocalLoss, MaskMetricLoss, MeanSimilarityLoss, TotalCodingRate, EntailmentLoss

class DecoderAssociator(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_heads, num_layers, object_dim, place_dim, 
                 num_img_patches, model, pr_loss, obj_loss, curv_init=40.0, learn_curv=True, **kwargs):
        super(DecoderAssociator, self).__init__()
        self.model_name = model
        self.object_dim = object_dim
        self.place_dim = place_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_img_patches = num_img_patches # 256 # 224//14 ** 2 [CLS]

        # Initialize curvature parameter for hyperbolic space. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }

        # Learnable scalars to ensure that object/place features have an expected
        # unit norm before exponential map (at initialization).
        self.object_alpha = nn.Parameter(torch.tensor(hidden_dim**-0.5).log())
        self.place_alpha = nn.Parameter(torch.tensor(hidden_dim**-0.5).log())
        
        # Learnable scalars for encoder outputs (place_enc and object_enc) after heads
        # Similar to MERU's visual_alpha and textual_alpha used in encode_image/encode_text
        self.place_enc_alpha = nn.Parameter(torch.tensor(output_dim**-0.5).log())
        self.object_enc_alpha = nn.Parameter(torch.tensor(output_dim**-0.5).log())

        # self.sep_token = nn.Parameter(torch.empty(1, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = hidden_dim,
            nhead = num_heads,
            dim_feedforward = int(hidden_dim * 4),
            dropout = 0.1,
            activation = 'gelu',
            layer_norm_eps = 1e-5,
            batch_first=True, 
            norm_first=False,
        )
        decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True) # 1e-5 or 1e-6?
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers, norm=decoder_norm)

        # box embedding
        self.box_emb = nn.Linear(4, hidden_dim, bias=False)
        self.whole_box = nn.Parameter(torch.tensor([0, 0, 224, 224], dtype=torch.float32), requires_grad=False)

        # input adaptor
        self.object_proj = nn.Linear(object_dim, hidden_dim, bias=False)
        self.place_proj = nn.Linear(place_dim, hidden_dim, bias=False)
        # output head
        # self.object_head = nn.Sequential(
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.GELU(approximate='tanh'),
        #     nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-5),
        #     nn.Linear(output_dim, output_dim),
        # )
        self.object_head = nn.Linear(hidden_dim, output_dim)
        
        # self.place_head = nn.Sequential(
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.GELU(approximate='tanh'),
        #     nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-5),
        #     nn.Linear(output_dim, output_dim),
        # )
        self.place_head = nn.Linear(hidden_dim, output_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_img_patches + 1, hidden_dim), requires_grad=False)

        self.measure_cos_pp = False
        if pr_loss == "bce":
            w = kwargs["pp_weight"] if "pp_weight" in kwargs else 1.0
            self.pr_loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([w]))
            self.measure_cos_pp = False
            self.pr_logit_scale = None
        elif pr_loss == "infonce":
            temperature = kwargs.get("temperature", 0.1)  # Default temperature if not provided
            self.pr_loss_fn = InfoNCELoss(temperature=temperature, learnable=False)
            self.measure_cos_pp = False
            # Initialize a learnable logit scale parameter (similar to MERU)
            self.pr_logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        else:
            self.pr_loss_fn = nn.MSELoss(reduction='none')
            self.measure_cos_pp = True
            self.pr_logit_scale = None

        self.measure_cos_obj = False
        if obj_loss == "bce":
            assert "pos_weight" in kwargs
            pos_weight = kwargs["pos_weight"]
            self.obj_loss_fn = MaskBCELoss(pos_weight=pos_weight)
            self.measure_cos_obj = False
            self.logit_scale = None
        elif obj_loss == "focal":
            assert "alpha" in kwargs
            assert "gamma" in kwargs
            self.obj_loss_fn = FocalLoss(alpha=kwargs["alpha"], gamma=kwargs["gamma"])
            self.measure_cos_obj = False
            self.logit_scale = None
        elif obj_loss == "infonce":
            temperature = kwargs.get("temperature", 0.1)  # Default temperature if not provided
            self.obj_loss_fn = InfoNCELoss(temperature=temperature, learnable=False)
            self.measure_cos_obj = False
            # Initialize a learnable logit scale parameter (similar to MERU)
            self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        else:
            self.obj_loss_fn = MaskMetricLoss()
            self.measure_cos_obj = True
            self.logit_scale = None
        self.obj_loss_fn_sim = MeanSimilarityLoss()
        self.obj_tcr = TotalCodingRate(eps=0.2)
        
        # Entailment loss: place should entail objects
        entail_weight = kwargs.get("entail", 1.0) # Default weight if not provided
        self.entail_weight = entail_weight
        if entail_weight > 0:
            aperture_threshold = kwargs.get("aperture_threshold", 1.0)
            self.entailment_loss_fn = EntailmentLoss(
                aperture_threshold=aperture_threshold,
                curv_init=curv_init,
                learn_curv=learn_curv
            )
        else:
            self.entailment_loss_fn = None

        self.initialize_weights()


    def initialize_weights(self,):
        
        self.apply(self._init_weights)

        grid_size = int(self.num_img_patches**.5)
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1], 
            grid_size=grid_size, 
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    # reference: from MAE's code base 
    # https://github.com/facebookresearch/mae/blob/main/models_mae.py#L68
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def pad_objects(self, object_emb):
        padded_object_emb = pad_sequence(object_emb, batch_first=True, padding_value=0)
        return padded_object_emb
    
    def get_query_mask(self, padded_object_emb):
        """
        Obtain masking for attention, since object embedding are padded.
        padded_object_emb: the real 0/1 masking
        """
        B, L, Ho = padded_object_emb.size()
        img_length = 1 # just 1 token for the whole-box query
        obj_mask = (padded_object_emb == 0).all(dim=-1).to(padded_object_emb.device)
        place_mask = torch.zeros(B, img_length, dtype = obj_mask.dtype, device = obj_mask.device)
        total_mask = torch.cat((place_mask, obj_mask), dim=1)
        return total_mask


    def forward(self, object_emb, place_emb, detections):
        """
        input:
            object_emb: list of B elements, each is a tensor of embeddings (K x Ho) of that image but in various lengths K.
            detectoins: list of B elements, each is a tensor of detections (K x 4) of that image, in various lengths K.
            place_emb: B x L x Hp, or B x H x W x Hp, D is the dimension of the place embeddings
        output:
            object_association_loss, place_recognition_loss
        """
        # pad object
   

        padded_obj_embd = self.pad_objects(object_emb)
        # print(padded_obj_embd.shape)
        B, K, Ho = padded_obj_embd.shape
        padded_obj_box = self.pad_objects(detections)
        
        whole_box_expanded = self.whole_box.unsqueeze(0).expand(B, 1, -1)
        
        query = torch.cat([whole_box_expanded, padded_obj_box], dim = 1) / 224.0 # hard-code nomalization
        # convert to embedding
        query = self.box_emb(query)

        query_mask = self.get_query_mask(padded_obj_embd) # B x K + 1 -> 1 for the whole_box preppended



        # flatten place
        if len(place_emb.size()) == 4:
            Hp = place_emb.size(1)
            place_emb = torch.einsum("bchw -> bhwc", place_emb)
            place_emb = place_emb.view(B, -1, Hp)

        # object and place embeddings, adapt to dimension
        object_feat = self.object_proj(padded_obj_embd) # B x K x D

        # place embeddings
        place_feat = self.place_proj(place_emb) # B x M x D
        
        # condition the query with embedding
        conditioning = torch.cat([place_feat.mean(dim=1, keepdim=True), object_feat], dim=1)
        query = query + conditioning

        memory = place_feat + self.pos_embed[:, :place_feat.size(1), :]

        # decoding
        decoded_emb = self.decoder(
            tgt = query,
            memory = memory,
            tgt_key_padding_mask = query_mask,
        )

        place_output = decoded_emb[:, 0, :]
        object_output = decoded_emb[:, 1:, :]

        # Clamp curvature and scaling factors (similar to MERU)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        
        # Clamp scaling factors for encoder outputs such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.place_enc_alpha.data = torch.clamp(self.place_enc_alpha.data, max=0.0)
        self.object_enc_alpha.data = torch.clamp(self.object_enc_alpha.data, max=0.0)
        
        # Clamp logit scale (similar to MERU) - ln(100) = ~4.6052
        if self.logit_scale is not None:
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        if self.pr_logit_scale is not None:
            self.pr_logit_scale.data = torch.clamp(self.pr_logit_scale.data, max=4.6052)

        # Pass decoder outputs through heads first (similar to MERU's encode_image/encode_text)
        place_enc = self.place_head(place_output) # B x output_dim
        object_enc = self.object_head(object_output) # B x K x output_dim

        # Map place_enc and object_enc to hyperbolic space after heads (similar to MERU)
        # Reshape for exp_map0
        B_place, H_place = place_enc.shape
        B_obj, K, H_obj = object_enc.shape
        
        # Apply scaling and exponential map to hyperbolic space for place_enc
        place_enc_flat = place_enc.reshape(-1, H_place)  # (B, H)
        with torch.autocast(device_type=place_enc.device.type, dtype=torch.float32):
            place_enc_flat = place_enc_flat * self.place_enc_alpha.exp()
            place_enc_flat = L.exp_map0(place_enc_flat, _curv)
        place_enc = place_enc_flat.reshape(B_place, H_place)
        
        # Apply scaling and exponential map to hyperbolic space for object_enc
        object_enc_flat = object_enc.reshape(-1, H_obj)  # (B*K, H)
        with torch.autocast(device_type=object_enc.device.type, dtype=torch.float32):
            object_enc_flat = object_enc_flat * self.object_enc_alpha.exp()
            object_enc_flat = L.exp_map0(object_enc_flat, _curv)
        object_enc = object_enc_flat.reshape(B_obj, K, H_obj)

        # Get curvature for predictions
        place_logits = self.predict_place(place_enc, _curv)
        object_logits = self.predict_object(object_enc, _curv)

        results = {
            'embeddings': object_enc,
            'place_embeddings': place_enc,
            'place_predictions': place_logits,
            'object_predictions': object_logits,
        }
        
        return results


    def predict_object(self, padded_obj_feat, curv):
        # object_embeddings: B x K x Ho, padded from object embedding (already in hyperbolic space)
        B, K, H = padded_obj_feat.size()
        # # object_predictions: BK x BK
        # Flatten for pairwise distance computation
        flatten_obj_feat = padded_obj_feat.view(-1, H)  # (B*K, H)
        
        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_obj_feats = dist.gather_across_processes(flatten_obj_feat)
        all_obj_feats = torch.cat(all_obj_feats, dim=0) # shape: (batch_size * world_size, embed_dim)

        # Compute hyperbolic pairwise distances (similar to MERU)
        # Use negative distance as similarity (smaller distance = higher similarity)
        with torch.autocast(device_type=padded_obj_feat.device.type, dtype=torch.float32):
            object_predictions = -L.pairwise_dist(flatten_obj_feat, all_obj_feats, curv)  # (BK, BK)
        return object_predictions  
        
    
    def predict_place(self, place, curv):
        # place embeddings already in hyperbolic space
        # place_predictions: B x B

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_pl_feats = dist.gather_across_processes(place)
        all_pl_feats = torch.cat(all_pl_feats, dim=0) # shape: (batch_size * world_size, embed_dim)

        # Compute hyperbolic pairwise distances (similar to MERU)
        # Use negative distance as similarity (smaller distance = higher similarity)
        with torch.autocast(device_type=place.device.type, dtype=torch.float32):
            place_logits = -L.pairwise_dist(place, all_pl_feats, curv)  # (B, B)
        return place_logits
    

    def get_loss(self, results, additional_info, match_inds, place_labels, weights):
        # prepare
        num_emb = results['embeddings'].size(1)
        reorderd_idx = get_match_idx(match_inds, additional_info, num_emb)
        logs = {}
        # get loss
        # object similarity loss with TCR regularizer
        sim_loss, mean_dis, tcr, id_counts = self.object_similarity_loss(results['embeddings'], reorderd_idx)
        logs['tcr'] = tcr.item()
        logs['obj_sim_loss'] = sim_loss.item()
        # logs['num_obj'] = id_counts.shape[0]
        logs['mean_dis'] = mean_dis.item()
        # logs['avg_num_instances'] = id_counts.sum().item() / (id_counts.shape[0] + 1e-5)

        # object_loss = weights['obj'] * sim_loss + weights['mean'] * mean_dis + weights['tcr'] * tcr
        # # object association loss
        object_loss = self.object_association_loss(results['object_predictions'], reorderd_idx)

        logs['running_loss_obj'] = object_loss.item()
        
        # place recognition loss
        place_loss = self.place_recognition_loss(results['place_predictions'], place_labels)

        # Initialize entailment_loss to 0 in case it's not computed
        # Use zeros_like to ensure it has the same device and requires_grad as object_loss
        entailment_loss = torch.zeros_like(object_loss)
        
        # Entailment loss: object should entail its place (similar to MERU)
        #print(self.entail_weight)
        if self.entail_weight > 0:
            # Get curvature (clamp it first, similar to forward method)
            self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
            _curv = self.curv.exp()
            
            place_enc = results['place_embeddings']  # B x h
            object_enc = results['embeddings']  # B x K x h
            
            # Flatten object_enc to get flatten_obj_feat (same as in predict_object)
            B, K, h = object_enc.shape
            flatten_obj_feat = object_enc.view(-1, h)  # (B*K) x h
            
            # Expand place_enc to match flatten_obj_feat: each object should entail its place
            # place_enc: B x h -> expand to (B*K) x h
            place_enc_expanded = place_enc.unsqueeze(1).expand(B, K, h)  # B x K x h
            place_enc_flat = place_enc_expanded.reshape(-1, h)  # (B*K) x h
            
            # Compute hyperbolic entailment loss: object should entail its place
            # Use autocast for higher precision (similar to MERU)
            with torch.autocast(device_type=place_enc.device.type, dtype=torch.float32):
                _angle = L.oxy_angle(flatten_obj_feat, place_enc_flat, _curv)
                #print(_angle)
                _aperture = L.half_aperture(flatten_obj_feat, _curv)
                #print(_aperture)
                entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()
        
        # Compute total loss with all components
        total_loss = weights['obj'] * object_loss + weights.get('pr', 0.0) * place_loss + weights.get('entail', 0.0) * entailment_loss
        logs['running_loss_pr'] = place_loss.item()
        logs['running_loss_entail'] = entailment_loss.item()
        
        # print(sim_loss, mean_dis, tcr, object_loss, place_loss)
        return total_loss, logs

    def place_recognition_loss(self, place_predictions, place_labels): # TODO: check implementation
        # place_predictions: B x (B * world_size) (similarity matrix with all GPUs)
        # place_labels: B x 1 (place labels for each sample in current batch)
        # loss: scalar
        
        # If using InfoNCELoss, need to convert place_labels to supervision_matrix
        if isinstance(self.pr_loss_fn, InfoNCELoss):
            B = place_labels.shape[0]
            # place_predictions is B x (B * world_size), but we only match within current batch
            # So we use only the first B columns for supervision
            place_predictions_local = place_predictions[:, :B]  # B x B
            
            # Convert place_labels (B x 1) to supervision_matrix (B x B)
            # supervision_matrix[i, j] = 1 if place_labels[i] == place_labels[j], else 0
            place_labels_expanded = place_labels.expand(B, B)  # B x B
            supervision_matrix = (place_labels_expanded == place_labels_expanded.T).float()  # B x B
            
            # Create mask (all positions are valid for place recognition within current batch)
            mask = torch.ones(B, B, device=place_predictions.device, dtype=torch.float32)
            
            # Apply logit scale if using InfoNCELoss (similar to MERU)
            if self.pr_logit_scale is not None:
                _scale = self.pr_logit_scale.exp()
                place_predictions_local = _scale * place_predictions_local
            
            loss = self.pr_loss_fn(place_predictions_local, supervision_matrix, mask)
        else:
            # For BCE or MSE loss, use the original implementation
            loss = self.pr_loss_fn(place_predictions, place_labels).mean()
        
        return loss
    
    def object_association_loss(self, object_predictions, reorderd_idx):
        """
        input:
            object_predictions: BN x BN, cosine similarity matrix
            reorder_idx: BN, padded, reordered gt_indices to match the pred_indices
        intermediate:
            supervision_matrix: BN x BN, binary matrix indicating the object association
            mask: BN x BN, binary matrix indicating the valid entries in the supervision_matrix
        output:
            loss: scalar
        """
        # supervision is already masked by the mask
        supervision_matrix, mask = get_association_sv(reorderd_idx)

        # Apply logit scale if using InfoNCELoss (similar to MERU)
        if self.logit_scale is not None:
            _scale = self.logit_scale.exp()
            object_predictions = _scale * object_predictions

        # using wrapped loss
        total_loss = self.obj_loss_fn(object_predictions, supervision_matrix, mask)
        return total_loss
    
    def object_similarity_loss(self, embeddings, matched_idx):
        """
        compute the similarity loss and the regularization
        """
        B, N, h = embeddings.size()
        flatten_embeddings = embeddings.view(-1, h)
        sim_loss, mean_dis_loss, id_counts = self.obj_loss_fn_sim(flatten_embeddings, matched_idx)
        tcr = self.obj_tcr(flatten_embeddings, matched_idx)
        return sim_loss, mean_dis_loss, tcr, id_counts
    

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
