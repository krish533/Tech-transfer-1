# Suggested Method References

These are the main references that support the modeling choices in the fine-tuned BERT pipeline.

## Core NLP model

Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL-HLT 2019.

Why cite it:

- establishes the `bert-base-uncased` architecture used for sentence classification
- standard citation for contextual transformer fine-tuning

Link:

- https://aclanthology.org/N19-1423/

## Probability calibration

Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. "On Calibration of Modern Neural Networks." ICML 2017.

Why cite it:

- standard reference for temperature scaling
- supports the post-hoc calibration step applied to BERT logits

Link:

- https://proceedings.mlr.press/v70/guo17a.html

## Legal-domain transformer support

Chalkidis, Ilias, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020. "LEGAL-BERT: The Muppets straight out of Law School." Findings of EMNLP 2020.

Why cite it:

- supports the use of transformer models on legal and policy text
- useful for motivating domain relevance even when the base model is general-purpose BERT

Link:

- https://aclanthology.org/2020.findings-emnlp.261/
