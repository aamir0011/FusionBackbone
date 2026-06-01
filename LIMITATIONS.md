# Limitations

## Missing-modality inference is not explicitly handled

The current IRENE reproduction pipeline assumes all modalities are available at inference time:
- image (`img`)
- chief-complaint text embedding (`cc` / `pdesc`)
- lab features (`lab` / `bts`)
- demographics (`sex`, `age` from `bics`)

### Why this is a limitation
- The model forward path expects all modality tensors to be present and shaped correctly.
- There is no explicit modality-missing mechanism (e.g., modality masks, learned missing tokens, gating, or fallback branches).
- If a modality is truly absent (e.g., `None` or missing record key), inference can fail unless placeholder values are injected.

### Current behavior in this workspace
- Data preparation scripts provide defaults/placeholders (e.g., zero lab vectors, default demographic values), which allows execution.
- This is not equivalent to robust missing-modality modeling; it is an input-completion workaround.

### Research impact
- Real clinical settings often have partial records.
- Therefore, model reliability and AUROC under missing-modality conditions are under-characterized.

### Recommended follow-up experiments
1. **Single-modality drop tests**: remove one modality at a time (image/text/lab/demo) and evaluate AUROC change.
2. **Multi-modality drop tests**: evaluate combinations of missing modalities.
3. **Explicit missing-modality modeling**:
   - modality-presence masks
   - learned "missing" embeddings/tokens
   - training-time modality dropout
4. **Report robustness curves**: AUROC vs. modality-availability scenarios.
