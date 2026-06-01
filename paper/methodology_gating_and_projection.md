## Methodology: Missing-Modality Learning via Modality-Wise Gating and Gating+Projection

To address incomplete multimodal inputs in medical visual question answering and diagnosis tasks, we investigate two model-side strategies: **modality-wise gating** and **gating+projection**. Both methods are built on the same multimodal backbone and differ in how they handle absent modalities at training and inference time.

### Problem Setup

Let each sample be represented by three modality streams:
- image features \(I\),
- text features \(T\),
- structured clinical features \(S\),

with corresponding availability indicators \(m = [m_I, m_T, m_S] \in \{0,1\}^3\).
The objective is multi-label disease prediction with classifier output \(\hat{y}\) and ground-truth labels \(y\).

---

### 1) Modality-Wise Gating

In modality-wise gating, a lightweight gating network \(G(\cdot)\) maps modality availability (and implicitly reliability) to continuous gate weights:
\[
[g_I, g_T, g_S] = G(m), \quad g_\bullet \in [0,1].
\]

Each modality stream is rescaled before multimodal fusion:
\[
\tilde{I} = g_I I, \quad \tilde{T} = g_T T, \quad \tilde{S} = g_S S.
\]

The fused representation is then used for downstream prediction. The training objective is standard multi-label classification loss:
\[
\mathcal{L}_{\text{gating}} = \mathcal{L}_{\text{cls}}(\hat{y}, y).
\]

**Interpretation.** Gating acts as a soft reliability filter: absent/noisy modalities are downweighted, while informative available modalities are emphasized.

---

### 2) Gating+Projection

Gating+projection extends gating with a projection (imputation-in-latent-space) module \(P(\cdot)\), which estimates missing modality embeddings from available context.

For each modality stream, projected representations are computed:
\[
[\hat{I}, \hat{T}, \hat{S}] = P(I,T,S,m).
\]

Final modality inputs are formed by gate-conditioned blending:
\[
I' = g_I I + (1-g_I)\hat{I},
\]
\[
T' = g_T T + (1-g_T)\hat{T},
\]
\[
S' = g_S S + (1-g_S)\hat{S}.
\]

This preserves observed information when present and substitutes projected information when absent.

The total loss combines classification with projection consistency:
\[
\mathcal{L}_{\text{gating+proj}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{proj}}\mathcal{L}_{\text{proj}},
\]
where \(\mathcal{L}_{\text{proj}}\) is reconstruction/projection loss in feature space and \(\lambda_{\text{proj}}\) controls regularization strength.

**Interpretation.** Unlike gating-only, this variant actively recovers missing modality content, improving robustness under severe missingness when projections are accurate.

---

### Practical Distinction

- **Gating**: reliability reweighting only (suppression/redistribution).
- **Gating+Projection**: reliability reweighting + latent modality completion.

In empirical analysis, this distinction typically creates a trade-off between full-condition discrimination and missing-condition robustness.

---

## Pseudo-code (Paper Format)

### Algorithm 1: Modality-Wise Gating

```text
Input: Training set D = {(I, T, S, y)}, missingness simulator M(·),
       multimodal model Fθ, gating network Gφ, epochs E
Output: Trained parameters θ, φ

for epoch = 1 to E do
    for minibatch B in D do
        # 1) Sample/obtain modality availability mask
        m ← M(B)                      # m = [mI, mT, mS]

        # 2) Compute gate values
        [gI, gT, gS] ← Gφ(m)

        # 3) Gate modality features
        I_tilde ← gI ⊙ I
        T_tilde ← gT ⊙ T
        S_tilde ← gS ⊙ S

        # 4) Forward + classification loss
        y_hat ← Fθ(I_tilde, T_tilde, S_tilde)
        L_cls ← BCE(y_hat, y)

        # 5) Optimize
        Update (θ, φ) using ∇(L_cls)
    end for
end for
return θ, φ
```

### Algorithm 2: Gating+Projection

```text
Input: Training set D = {(I, T, S, y)}, missingness simulator M(·),
       multimodal model Fθ, gating network Gφ, projector Pψ,
       projection weight λproj, epochs E
Output: Trained parameters θ, φ, ψ

for epoch = 1 to E do
    for minibatch B in D do
        # 1) Sample/obtain modality availability mask
        m ← M(B)                      # m = [mI, mT, mS]

        # 2) Compute gate values
        [gI, gT, gS] ← Gφ(m)

        # 3) Predict projected modality representations
        [I_hat, T_hat, S_hat] ← Pψ(I, T, S, m)

        # 4) Gate-conditioned blending (observed + projected)
        I_prime ← gI ⊙ I + (1-gI) ⊙ I_hat
        T_prime ← gT ⊙ T + (1-gT) ⊙ T_hat
        S_prime ← gS ⊙ S + (1-gS) ⊙ S_hat

        # 5) Forward + joint objective
        y_hat ← Fθ(I_prime, T_prime, S_prime)
        L_cls  ← BCE(y_hat, y)
        L_proj ← ProjectionLoss([I_hat, T_hat, S_hat], [I, T, S])
        L_total ← L_cls + λproj · L_proj

        # 6) Optimize
        Update (θ, φ, ψ) using ∇(L_total)
    end for
end for
return θ, φ, ψ
```
