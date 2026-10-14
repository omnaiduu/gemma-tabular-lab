# Engineering blog outline — evidence pending

Working title: **Predicting prices with Gemma and categorical digit heads**.

1. The business task and available information at prediction time.
2. The user's question about turning numerical prediction into small choices.
3. Scientific notation, finite precision, zero handling, and exponent errors.
4. Gemma's backbone, logits, LoRA, and parallel output heads.
5. Why the direct-regression head is a necessary control, and its attribution.
6. Strong traditional baselines with and without text features.
7. Data provenance, split isolation, model/version pinning, and budget constraints.
8. Measured results: numerical error, severe mistakes, latency, memory, and cost.
9. Failed attempts, negative results, and what changed in the implementation.
10. Limits of the public benchmark and the next test on fresh company data.

Do not fill the results section with hypothetical measurements. Every table
should reference a saved run report and the corresponding experiment settings.
Model artifacts and source data are not part of the public repository.
