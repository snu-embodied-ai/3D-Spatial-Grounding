import torch
import torch.nn as nn

import os, sys
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
sys.path.extend([ROOT_DIR, CUR_DIR])

from .Uni3D.models.uni3d import create_uni3d
from .modules.transformer import BiDirectionalTransformerDecoder, BiDirectionalTransformerEncoder
from .modules.map_generator import OccupancyMapGenerator, HeatMapGenerator


class Spatial3D_lang(nn.Module):
    def __init__(self, cfg):
        """
        Freeze the pretrained encoders first!
        """
        # TODO
        # 1. Extract the embedding of the whole scene
        # 2. Extract the embedding of each region
        # 3. Fuse the two embeddings (scene + regions) -> Self Attetion & Cross Attention
        # 4. Cross modal fusion (Text + 3D visual information)
        # 5. Decoder - output heatmap

        # MAKE MODULES EASY TO DECOMPOSE AND COMBINE
        # TO DO OTHER EXPERIMENTS EASIER
        # ALSO EASIER TO CHANGE THE ARCHITECTURE (e.g. integrating LLM, introducing other learning framework)

        super(Spatial3D_lang, self).__init__()


        # 0. Setting Hyperparmeters & Variables
        self.use_whole_scene = cfg["use_whole_scene"]
        
        self.encoded_hidden_dim = cfg["CLIP"]["hidden_dim"]
        self.lang_num_tokens = cfg["CLIP"]["context_length"]
        self.encode_text_in_cpu = cfg["CLIP"]["to_CPU"]
        
        self.use_embed2trans = cfg["DINO"]["use_embed2trans_projector"]
        self.use_shared_embed2trans = cfg["DINO"]["use_shared_embed2trans_projector"]
        self.vis_pos_enc_type = cfg["DINO"]["vision_position_encoding"]
        self.lang_pos_enc_type = cfg["DINO"]["text_position_encoding"]

        self.transformer_input_dim = cfg["DINO"]["hidden_dim"]
        self.transformer_hidden_dim = cfg["DINO"]["dim_feedforward"]
        
        self.transformer_num_heads = cfg["DINO"]["num_heads"]
        self.transformer_layer_num = cfg["DINO"]["num_biformers"]
        self.transformer_dropout = cfg["DINO"]["dropout"]
        self.transformer_pre_norm = cfg["DINO"]["pre_norm"]

        # DINO - Query related hyperparameters
        self.num_queries = cfg["DINO"]["Queries"]["num_queries"]

        # Task head - hyperparameters for heatmap generation

        # 1. Load 3D point cloud encoders
        # TODO : Also implement a RGBD version
        if self.use_whole_scene:
            # Create Scene Encoder
            pass

        self.region_encoder = create_uni3d(cfg["uni3d"])

        checkpoint = torch.load(cfg["uni3d"]["ckpt_path"])
        sd = checkpoint['module']
        if not cfg["distributed"] and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        self.region_encoder.load_state_dict(sd)


        # 2. Learnable Queries/Position & Projectors
        # 2-1. Projectors
        if self.use_embed2trans and not self.use_shared_embed2trans:
            self.vis_projector = nn.Sequential(
                nn.Linear(self.encoded_hidden_dim, self.encoded_hidden_dim),
                nn.GELU(),
                nn.Linear(self.encoded_hidden_dim, self.transformer_input_dim)
            )

            self.lang_projector = nn.Sequential(
                nn.Linear(self.encoded_hidden_dim, self.encoded_hidden_dim),
                nn.GELU(),
                nn.Linear(self.encoded_hidden_dim, self.transformer_input_dim)
            )

        elif self.use_embed2trans and self.use_shared_embed2trans:
            self.shared_projector = nn.Sequential(
                nn.Linear(self.encoded_hidden_dim, self.encoded_hidden_dim),
                nn.GELU(),
                nn.Linear(self.encoded_hidden_dim, self.transformer_input_dim)
            )
        
        # 2-2. Positional Encoding
        if self.vis_pos_enc_type == "learnable":
            self.vis_pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.transformer_input_dim)
            )

        if self.lang_pos_enc_type == "learnable":
            self.lang_pos_embed = nn.Embedding(self.lang_num_tokens, self.transformer_input_dim)

        # 2-3. Queries
        self.query_pos = nn.Embedding(self.num_queries, self.transformer_input_dim)

        # 3. Load Bidirectional Transformer Encoder
        self.encoder = BiDirectionalTransformerEncoder(
            v_dim=self.transformer_input_dim,
            l_dim=self.transformer_input_dim,
            embed_dim=self.transformer_hidden_dim,
            num_heads=self.transformer_num_heads,
            num_biformers=self.transformer_layer_num,
            pre_norm=self.transformer_pre_norm,
            dropout=self.transformer_dropout,
            use_whole_scene=self.use_whole_scene
        )
        # 4. Load Bidirectional Transformer Decoder
        self.decoder = BiDirectionalTransformerDecoder(
            v_dim=self.transformer_input_dim,
            l_dim=self.transformer_input_dim,
            embed_dim=self.transformer_hidden_dim,
            num_heads=self.transformer_num_heads,
            num_biformers=self.transformer_layer_num,
            pre_norm=self.transformer_pre_norm,
            dropout=self.transformer_dropout,
            use_whole_scene=self.use_whole_scene
        )

        # 5. Task head for generating occupancy map
        self.occupany_map_generator = OccupancyMapGenerator(
            embedding_dim=self.transformer_hidden_dim,
            nhead=self.transformer_num_heads,
            activation="gelu",
            pre_norm=self.transformer_pre_norm,
            taskhead_cfg=cfg["OccupancyHead"],
            dropout=self.transformer_dropout)

        # 5. Task head for mapping the candidate queries to heatmap output
        self.heatmap_generator = HeatMapGenerator(
            nhead=1,
            hidden_dim=self.transformer_hidden_dim,
            activation="gelu",
            pre_norm=self.transformer_pre_norm,
            taskhead_cfg=cfg["HeatmapHead"],
            dropout=self.transformer_dropout)
        

    def freeze_pretrained_modules(self):
        """
        Freezing pretrained modules (3D point cloud encoder and Text encoder)
        and returning the trainable parameters
        """
        for module in [self.region_encoder.parameters()]:
            for param in module:
                param.requires_grad = False

        trainable_params = [p for p in self.parameters() if p.requires_grad]

        return trainable_params


    def forward(self, vision_dict, language_dict):
        """
        vision_dict : Vision data dict
        langauge_dict: Language data dict - language descriptions are already tokenized before the forward operation
        """    

        # 1. Extract Vision embedding (Point cloud) from pretrained Uni3D
        vision_dict = self.region_encoder(vision_dict)

        # 2. Project the embeddings to the feature space of the fusion module
        if self.use_whole_scene:
            pass
        else:
            if self.use_embed2trans:
                if self.use_shared_embed2trans:
                    # Concatenate vision and language embeddings to share the projector weights
                    vis_embed = vision_dict["region_embedding"]
                    lang_embed = language_dict["input_ids"]
                    num_vis_groups = vis_embed.size()[1]
                    # num_text_tokens = lang_embed.size()[1]
                    cat_embed = torch.cat([vis_embed, lang_embed], dim=1)

                    proj_cat_embed = self.shared_projector(cat_embed)
                    vision_dict["region_embedding"] = proj_cat_embed[:,:num_vis_groups,:]
                    language_dict["input_ids"] = proj_cat_embed[:,num_vis_groups:,:]
                else:
                    # Project each modality using separate projectors
                    proj_vis_embed = self.vis_projector(vision_dict["region_embedding"])
                    proj_lang_embed = self.lang_projector(language_dict["input_ids"])

                    vision_dict["region_embedding"] = proj_vis_embed
                    language_dict["input_ids"] = proj_lang_embed

        # 3. Get Vision / Language Positional Embeddings
        if self.vis_pos_enc_type == "learnable":
            vision_dict["region_position_encoding"] = self.vis_pos_embed(vision_dict["centers"])
        else:
            # For sine encoding and other encoding
            pass

        if self.lang_pos_enc_type == "learnable":
            language_dict["text_position_encoding"] = self.lang_pos_embed(language_dict["position_ids"])
        else:
            # For sine encoding and other encoding
            pass
        
        # 4. Pass the encoder
        vision_dict, language_dict = self.encoder(vision_dict, language_dict)

        # 5. Generate Occupancy map
        occupancy_map = self.occupany_map_generator(vision_dict["heatmap_grid_centers"],
                                                    vision_dict["region_embedding"],
                                                    vision_dict["region_position_encoding"])
        vision_dict["output_occupancy_map"] = occupancy_map
        # print("OCCUPANCY!!!!\n")
        # print(occupancy_map[0])
        # print(occupancy_map.max(), occupancy_map.min())


        # 6. Initialize the Queries
        # TODO: Create Query Tokens as much as the output heatmap size. E.g. If the heatmap should be 50*60, then create 300 query tokens which corresponds to one voxel(pixel, region)

        cur_device = vision_dict["centers"].device
        B = vision_dict["centers"].size(0)
        queries = torch.zeros((B, self.num_queries, self.transformer_hidden_dim)).to(cur_device)
        vision_dict["query_embedding"] = queries
        vision_dict["query_position_encoding"] = self.query_pos.weight

        # 6. Pass the decoder
        vision_dict, language_dict = self.decoder(vision_dict, language_dict)

        # 7. Generate Heatmap
        heatmap = self.heatmap_generator(vision_dict["heatmap_grid_centers"],
                                         vision_dict["query_embedding"],
                                         vision_dict["query_position_encoding"])
        vision_dict["output_heatmap"] = heatmap
        # print("HEATMAP!!!!\n")
        # print(heatmap[0])
        # print(heatmap.max(), heatmap.min())

        # 8. Return Final Output - The query outputs are logits
        return vision_dict, language_dict