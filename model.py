import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MFSC_DTI(nn.Module):
    def __init__(self, hp,protein_MAX_LENGH=1000,drug_MAX_LENGH=100):
        super(MFSC_DTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
                                  self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
                                     self.protein_kernel[0] - self.protein_kernel[1] - \
                                     self.protein_kernel[2] + 3
        self.feature_size = 164  # 确认的特征数量

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)


        self.drug_bert_fc = nn.Linear(768, 160)
        self.protein_bert_fc = nn.Linear(1024, 160)

        self.classifier = Classifier(
            cnn_dim=320,  # final_drug_feat.mean + final_protein_feat.mean
            feature_dim=2891,  # drug_fused_feat + protein_fused_feat
            bert_dim=320  # molecules_smiles_LM + proteins_acids_LM
        )

    def forward(self, drug, protein, morgan_fp, avalon_fp, maccs_fp, protein_features, drug_bert, protein_bert):

        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        cnn_pair = torch.cat([drugConv,  proteinConv], dim=1)

        feature_pair = torch.cat([morgan_fp, avalon_fp, maccs_fp,protein_features], dim=1)

        # 处理药物Bert特征
        drug_bert_feat = self.drug_bert_fc(drug_bert)

        # 处理蛋白质Bert特征
        protein_bert_feat = self.protein_bert_fc(protein_bert)

        # 将Bert特征与原始特征融合
        bert_pair = torch.cat([drug_bert_feat, protein_bert_feat], dim=1)

        # 分类层
        final_pred, pred_combined, pred_fusion, pred_bert = self.classifier(
            cnn_pair, feature_pair, bert_pair
        )

        return final_pred, pred_combined, pred_fusion, pred_bert


class MultiChannelSharedCrossAttention(nn.Module):
    """
    多通道共享权重交叉注意力机制
    支持三类特征的两两交互和融合
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # 共享的权重矩阵
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WO = nn.Linear(d_model, d_model)

        # 融合权重（可学习）
        self.fusion_weights = nn.Parameter(torch.ones(3, 3))  # 3x3的融合权重矩阵

    def _single_attention(self, query_feat, key_value_feat):
        """单个注意力计算"""
        batch_size = query_feat.size(0)

        Q = self.WQ(query_feat)
        K = self.WK(key_value_feat)
        V = self.WV(key_value_feat)

        # 多头重塑
        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # 注意力计算
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        Z = torch.matmul(attn, V)

        # 合并多头
        Z = Z.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.WO(Z)

    def forward(self, feat_A, feat_B, feat_C):
        """
        三类特征的两两交互

        Args:
            feat_A: 特征A [batch, d_model]
            feat_B: 特征B [batch, d_model]
            feat_C: 特征C [batch, d_model]

        Returns:
            enhanced_A, enhanced_B, enhanced_C: 增强后的特征
        """
        # 添加序列维度 [batch, d_model] -> [batch, 1, d_model]
        feat_A = feat_A.unsqueeze(1)
        feat_B = feat_B.unsqueeze(1)
        feat_C = feat_C.unsqueeze(1)

        # A与其他特征的交互
        A_attn_B = self._single_attention(feat_A, feat_B)  # A关注B
        A_attn_C = self._single_attention(feat_A, feat_C)  # A关注C

        # B与其他特征的交互
        B_attn_A = self._single_attention(feat_B, feat_A)  # B关注A
        B_attn_C = self._single_attention(feat_B, feat_C)  # B关注C

        # C与其他特征的交互
        C_attn_A = self._single_attention(feat_C, feat_A)  # C关注A
        C_attn_B = self._single_attention(feat_C, feat_B)  # C关注B

        # 移除序列维度 [batch, 1, d_model] -> [batch, d_model]
        A_attn_B = A_attn_B.squeeze(1)
        A_attn_C = A_attn_C.squeeze(1)
        B_attn_A = B_attn_A.squeeze(1)
        B_attn_C = B_attn_C.squeeze(1)
        C_attn_A = C_attn_A.squeeze(1)
        C_attn_B = C_attn_B.squeeze(1)
        feat_A = feat_A.squeeze(1)
        feat_B = feat_B.squeeze(1)
        feat_C = feat_C.squeeze(1)

        # 加权融合（使用可学习的权重）
        w = F.softmax(self.fusion_weights, dim=1)

        enhanced_A = (w[0, 0] * feat_A +
                      w[0, 1] * A_attn_B +
                      w[0, 2] * A_attn_C)

        enhanced_B = (w[1, 0] * feat_B +
                      w[1, 1] * B_attn_A +
                      w[1, 2] * B_attn_C)

        enhanced_C = (w[2, 0] * feat_C +
                      w[2, 1] * C_attn_A +
                      w[2, 2] * C_attn_B)

        return enhanced_A, enhanced_B, enhanced_C


class Classifier(nn.Module):
    def __init__(self, cnn_dim, feature_dim, bert_dim, hidden_dim=512):
        super(Classifier, self).__init__()

        # 特征细化模块
        self.refine_cnn = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.refine_feature = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.refine_bert = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1))

        # 初步预测分支
        self.pred_cnn = nn.Linear(hidden_dim, 2)
        self.pred_feature = nn.Linear(hidden_dim, 2)
        self.pred_bert = nn.Linear(hidden_dim, 2)

        # 修正：添加一个线性层来处理注意力融合后的特征，使其维度与concat_fused一致
        self.attention_fusion_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 输入是 3 * hidden_dim
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.cross_attention_fusion = MultiChannelSharedCrossAttention(
            d_model=hidden_dim,
            num_heads=4
        )

        # 这个层保留，用于处理拼接融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1))

        # 最终预测
        self.final_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2))

    def forward(self, cnn_pair, features_pair, bert_pair):
        # 1. 特征细化
        refined_cnn = self.refine_cnn(cnn_pair)
        refined_feature = self.refine_feature(features_pair)
        refined_bert = self.refine_bert(bert_pair)

        # 2. 初步预测
        pred_cnn = self.pred_cnn(refined_cnn)
        pred_feature = self.pred_feature(refined_feature)
        pred_bert = self.pred_bert(refined_bert)

        # 3. 特征融合
        # 交叉注意力融合
        fused_cnn, fused_feature, fused_bert = self.cross_attention_fusion(
            refined_cnn, refined_feature, refined_bert
        )

        # 直接拼接融合后的特征
        fused_features = torch.cat([fused_cnn, fused_feature, fused_bert], dim=-1)
        fused_features = self.attention_fusion_adapter(fused_features)  # [batch_size, hidden_dim]

        # 拼接融合（原始特征）
        concat_features = torch.cat([refined_cnn, refined_feature, refined_bert], dim=-1)
        concat_fused = self.fusion_layer(concat_features)  # [batch_size, hidden_dim]

        # 4. 最终融合预测
        final_features = fused_features + concat_fused
        final_pred = self.final_predictor(final_features)

        return final_pred, pred_cnn, pred_feature, pred_bert
