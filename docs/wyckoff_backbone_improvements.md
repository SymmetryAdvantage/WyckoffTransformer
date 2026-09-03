# Deep-Dive: Upgrading the Core Wyckoff-Space Backbone

This document details the mathematical formulations, architectural designs, and concrete PyTorch implementations for three core enhancements to the **Wyckoff Transformer (WyFormer)** backbone:

1. **Crystallographic & Chemical Pairwise Relational Attention Biases**
2. **Learned Set Pooling (Pooling by Multihead Attention – PMA)**
3. **Logit Z-Loss Regularization**

---

## 1. Crystallographic & Chemical Pairwise Relational Attention Biases

### 1.1 Motivation & Theoretical Basis
In crystal structure generation, atomic sites form an **unordered set** $\mathcal{S} = \{ (e_i, ss_i, enum_i) \}_{i=1}^N$ in a given space group $G$. Standard Transformers use 1D sequence positional encodings (e.g., sinusoidal, learned 1D, or RoPE), which break permutation invariance unless sequences are randomly permuted during training.

However, treating all site pairs as isotropic ignores essential physical and group-theoretic structure:
- **Chemical Relations**: Elements exhibit strong pairwise preferences dictated by electronegativity differences $\Delta \chi_{ij} = |\chi_i - \chi_j|$ and ionic/covalent radius ratios $r_i / r_j$ (e.g., Pauling rules, Goldschmidt tolerance factors).
- **Crystallographic / Wyckoff Symmetry Relations**: Within space group $G$, pairs of Wyckoff positions $(WP_i, WP_j)$ possess distinct geometric compatibility constraints, shared symmetry subgroups, and minimum distance thresholds.

### 1.2 Mathematical Formulation
Instead of positional encodings, we inject a **permutation-invariant relational bias** $B_{ij}$ directly into the scaled dot-product attention logits:

$$\mathcal{A}_{ij} = \frac{\text{RMSNorm}(Q_i) \cdot \text{RMSNorm}(K_j)^T}{\sqrt{d_k}} + B_{\text{chem}}(e_i, e_j) + B_{\text{wyckoff}}(G, ss_i, ss_j)$$

where:

1. **Chemical Relational Bias $B_{\text{chem}}(e_i, e_j)$**:
   $$B_{\text{chem}}(e_i, e_j) = W_e^\top \left( \mathbf{v}_{e_i} \odot \mathbf{v}_{e_j} \right) + \text{MLP}_{\text{chem}}\left( [\Delta \chi_{ij}, |r_i - r_j|, r_i / r_j] \right)$$
   - $\mathbf{v}_{e} \in \mathbb{R}^{d_{\text{elem}}}$: Learned element embedding vector.
   - $\Delta \chi_{ij}$: Pauling electronegativity difference.
   - $r_i, r_j$: Ionic / covalent radii.
   - $B_{\text{chem}}$ is symmetric: $B_{\text{chem}}(e_i, e_j) = B_{\text{chem}}(e_j, e_i)$.

2. **Wyckoff Symmetry Relational Bias $B_{\text{wyckoff}}(G, ss_i, ss_j)$**:
   $$B_{\text{wyckoff}}(G, ss_i, ss_j) = \mathbf{E}_{\text{sg}}(G) + \mathbf{E}_{\text{pair}}(ss_i, ss_j)$$
   - $\mathbf{E}_{\text{pair}} \in \mathbb{R}^{N_{ss} \times N_{ss} \times N_{\text{heads}}}$: Symmetric pairwise site-symmetry interaction matrix.

### 1.3 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CrystallographicRelationalAttention(nn.Module):
    """
    Multi-Head Attention with QK-Norm and Crystallographic/Chemical Relational Biases.
    Preserves exact set-permutation invariance across Wyckoff sites.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_elements: int = 100,
        num_site_symmetries: int = 80,
        chem_feat_dim: int = 3,  # [delta_electronegativity, radius_diff, radius_ratio]
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # QK-Norm
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)

        # Pairwise Chemical Bias
        self.elem_embeddings = nn.Embedding(num_elements, self.d_k)
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, nhead)
        )

        # Pairwise Site-Symmetry Bias
        self.ss_pair_bias = nn.Embedding(num_site_symmetries * num_site_symmetries, nhead)
        self.num_ss = num_site_symmetries

        self.dropout = nn.Dropout(dropout)

    def compute_relational_bias(
        self,
        elements: torch.Tensor,         # [B, N]
        site_symmetries: torch.Tensor,  # [B, N]
        chem_features: Optional[torch.Tensor] = None # [B, N, N, chem_feat_dim]
    ) -> torch.Tensor:
        """Computes [B, nhead, N, N] pairwise attention bias."""
        B, N = elements.shape

        # 1. Chemical element interaction bias
        # Dot-product of element embeddings per head
        elem_emb = self.elem_embeddings(elements) # [B, N, d_k]
        elem_bias = torch.einsum("bid,bjd->bij", elem_emb, elem_emb) # [B, N, N]
        elem_bias = elem_bias.unsqueeze(1).expand(-1, self.nhead, -1, -1) # [B, H, N, N]

        if chem_features is not None:
            # Add continuous physical feature bias: [B, N, N, chem_feat_dim] -> [B, N, N, H]
            phys_bias = self.chem_mlp(chem_features).permute(0, 3, 1, 2) # [B, H, N, N]
            elem_bias = elem_bias + phys_bias

        # 2. Site symmetry pair bias
        ss_i = site_symmetries.unsqueeze(2).expand(-1, -1, N) # [B, N, N]
        ss_j = site_symmetries.unsqueeze(1).expand(-1, N, -1) # [B, N, N]
        pair_idx = ss_i * self.num_ss + ss_j                  # [B, N, N]
        ss_bias = self.ss_pair_bias(pair_idx).permute(0, 3, 1, 2) # [B, H, N, N]

        return elem_bias + ss_bias

    def forward(
        self,
        x: torch.Tensor,                # [B, N, d_model]
        elements: torch.Tensor,         # [B, N]
        site_symmetries: torch.Tensor,  # [B, N]
        chem_features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None # [B, N], True for padding
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # Project and reshape to [B, nhead, N, d_k]
        Q = self.q_proj(x).view(B, N, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.nhead, self.d_k).transpose(1, 2)

        # Apply QK-Norm
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Scaled dot-product attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Inject Relational Biases
        rel_bias = self.compute_relational_bias(elements, site_symmetries, chem_features)
        attn_scores = attn_scores + rel_bias

        # Apply padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, N] -> [B, 1, 1, N]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V) # [B, nhead, N, d_k]
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)
```

---

## 2. Learned Set Pooling (Pooling by Multihead Attention – PMA)

### 2.1 Motivation
In `CascadeTransformer`, set aggregation across variable numbers of occupied Wyckoff sites is computed using static operations:
- `sum`: Sensitive to sequence length; sums grow unbounded for complex unit cells.
- `mean`: Treats tiny trace sites (e.g. 1a) with the same weight as high-multiplicity sites (e.g. 48h).
- `max`: Discards additive stoichiometric information.

```
Current:    [Site 1]  [Site 2]  ...  [Site K]  ──►  Uniform Sum / Mean (Fixed heuristic)

Proposed:   [Site 1]  [Site 2]  ...  [Site K]  (Keys / Values)
               ▲         ▲              ▲
               └─────────┼──────────────┘
                         │   Cross-Attention
               ┌─────────┴──────────────┐
               │ Seed Queries S_1..k    │ ──►  Context-Aware Global Crystal Latent Z
               └────────────────────────┘
```

### 2.2 Mathematical Formulation
We adopt **Pooling by Multihead Attention (PMA)** from Set Transformer theory. Let $H \in \mathbb{R}^{N \times d}$ be the set of site embeddings and $S \in \mathbb{R}^{k \times d}$ be $k$ learnable query seed vectors (where $k=1$ for a single global crystal token, or $k=4$ for multi-faceted representations):

$$\text{PMA}_k(H) = \text{MAB}(S, H) = \text{MultiHeadAttn}\left(Q = S, K = H, V = H\right)$$

Following PMA, an optional **Induced Set Attention Block (ISAB)** or Multi-Layer Perceptron processes the pooled representation:
$$Z_{\text{global}} = \text{LayerNorm}\left( S + \text{MAB}(S, H) \right)$$
$$Z_{\text{global}} = \text{FFN}(Z_{\text{global}})$$

*Properties*:
1. **Strict Permutation Invariance**: Permuting the rows of $H$ yields the exact same pooled output $Z_{\text{global}}$.
2. **Adaptive Attention**: The model learns to attend more strongly to high-multiplicity sites, dominant anions, or coordinating cations.

### 2.3 PyTorch Implementation

```python
class PoolingByMultiheadAttention(nn.Module):
    """
    Learned Set Pooling (PMA) module for aggregating variable-length Wyckoff site sets
    into a fixed-size, permutation-invariant global crystal vector.
    """
    def __init__(self, d_model: int, nhead: int, num_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_seeds = num_seeds
        self.d_model = d_model
        
        # Learnable seed queries: S in R^{num_seeds x d_model}
        self.seed_queries = nn.Parameter(torch.randn(num_seeds, d_model))
        nn.init.xavier_uniform_(self.seed_queries)

        self.mab = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm_seed = nn.RMSNorm(d_model)
        self.norm_sites = nn.RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        site_embeddings: torch.Tensor,        # [B, N_sites, d_model]
        key_padding_mask: Optional[torch.Tensor] = None # [B, N_sites], True for pad
    ) -> torch.Tensor:
        """
        Returns:
            [B, d_model] if num_seeds == 1, else [B, num_seeds * d_model]
        """
        B = site_embeddings.size(0)

        # Expand seed queries across batch: [B, num_seeds, d_model]
        seeds = self.seed_queries.unsqueeze(0).expand(B, -1, -1)

        # Query = Seeds, Keys/Values = Site Embeddings
        norm_seeds = self.norm_seed(seeds)
        norm_sites = self.norm_sites(site_embeddings)

        pooled, _ = self.mab(
            query=norm_seeds,
            key=norm_sites,
            value=norm_sites,
            key_padding_mask=key_padding_mask
        ) # [B, num_seeds, d_model]

        # Residual + FFN
        out = seeds + pooled
        out = out + self.ffn(out)

        if self.num_seeds == 1:
            return out.squeeze(1) # [B, d_model]
        return out.flatten(start_dim=1) # [B, num_seeds * d_model]
```

---

## 3. Logit Z-Loss Regularization

### 3.1 Problem: Unbounded Logit Drift in Discrete Generation
In multiclass next-token cross-entropy training:
$$\mathcal{L}_{\text{CE}} = - \log \frac{\exp(\ell_{y})}{\sum_{j=1}^V \exp(\ell_j)} = -\ell_y + \log \sum_{j=1}^V \exp(\ell_j)$$

Notice that $\mathcal{L}_{\text{CE}}$ is invariant to adding an arbitrary constant $C$ to all logits: $\ell_j \leftarrow \ell_j + C$. 

During extended training (especially with adaptive optimizers like AdamW or Schedule-Free), logits can drift to large absolute values ($|\ell_j| > 10^3$). This causes:
1. **Softmax gradient vanishing / instability**.
2. **Numerical overflow / underflow** when casting to `torch.bfloat16` or `torch.float16`.
3. **Severe calibration degradation**: Unscaled large logits destroy temperature calibration during autoregressive generation.

### 3.2 Formulation & Derivation
**Z-loss** penalizes the partition function $\log Z = \log \sum_{j=1}^V \exp(\ell_j)$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + c_z \cdot \mathcal{L}_z$$
$$\mathcal{L}_z = \left( \log \sum_{j=1}^V \exp(\ell_j) \right)^2$$

where $c_z$ is a small weighting factor (typically $10^{-4}$).

#### Gradient Analysis:
$$\frac{\partial \mathcal{L}_z}{\partial \ell_i} = 2 \left( \log \sum_{j=1}^V \exp(\ell_j) \right) \cdot \frac{\exp(\ell_i)}{\sum_{j=1}^V \exp(\ell_j)} = 2 (\log Z) \cdot P(y = i)$$

- When $\log Z > 0$ (logits are systematically large and positive), $\frac{\partial \mathcal{L}_z}{\partial \ell_i} > 0$, pulling logits down.
- When $\log Z < 0$ (logits are excessively negative), the gradient pushes logits up toward 0.
- Softmax probabilities $P(y=i)$ are preserved, but logits stay bounded around 0 with zero runtime overhead.

### 3.3 PyTorch Implementation

```python
class CrossEntropyWithZLoss(nn.Module):
    """
    Cross Entropy Loss with auxiliary Logit Z-Loss regularization.
    Prevents logit scale drift and stabilizes FP16/BF16 mixed-precision training.
    """
    def __init__(self, z_loss_weight: float = 1e-4, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.z_loss_weight = z_loss_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] raw unnormalized model predictions.
            targets: [B] ground-truth class indices.
        """
        # Standard Cross-Entropy
        ce_loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )

        if self.z_loss_weight <= 0.0:
            return ce_loss

        # Log-sum-exp: log(sum(exp(logits)))
        log_z = torch.logsumexp(logits, dim=-1) # [B]
        z_loss = torch.square(log_z)            # [B]

        if self.reduction == "mean":
            z_loss = z_loss.mean()
        elif self.reduction == "sum":
            z_loss = z_loss.sum()

        return ce_loss + self.z_loss_weight * z_loss
```

---

## 4. Integration Blueprint for WyckoffTransformer Codebase

### Mapping to Existing Modules

```
WyckoffTransformer Repo
│
├── cascade/model.py
│   ├── Replace nn.TransformerEncoderLayer  ──►  AdaLNZeroRelationalTransformerLayer
│   │                                            (QK-Norm + Relational Attention + SwiGLU)
│   └── In CascadeTransformer.forward()     ──►  Replace sum/mean aggregation with PMA
│
├── trainer.py
│   └── In WyckoffTrainer.get_loss()        ──►  Replace criterion with CrossEntropyWithZLoss
│
└── generator.py
    └── In WyckoffGenerator.generate()      ──►  Utilize stabilized logits & PMA global states
```

### Summary of Benefits

| Enhancement | Problem Solved | Core Advantage |
| :--- | :--- | :--- |
| **Relational Attention Biases** | Isotropic attention ignores physics & group theory | Embeds pairwise chemical compatibility & Wyckoff subgroup constraints without violating permutation invariance. |
| **Learned Set Pooling (PMA)** | Fixed sum/mean ignores site multiplicity & role | Dynamic attention-based global crystal aggregation; produces richer context vectors for autoregression. |
| **Logit Z-Loss** | Large logit drift causing instability in BF16 | Enforces bounded logit scales ($\log Z \approx 0$); stabilizes gradient norms and improves temperature calibration. |
