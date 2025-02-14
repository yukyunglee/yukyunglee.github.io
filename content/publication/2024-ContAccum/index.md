---
title: "A Gradient Accumulation Method for Dense Retriever under Memory Constraint"
authors:
- Jaehee Kim
- admin
- Pilsung Kang
author_notes:
- ''
- ''
- ''
- ''
- ''
date: "2024-09-18T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2024-09-18T00:00:00Z"

# Publication type.
publication_types: ["article"]

# Publication name and optional abbreviated publication name.
publication: "NeurIPS 2024"
abstract: InfoNCE loss is commonly used to train dense retriever in information retrieval tasks. It is well known that a large batch is essential to stable and effective training with InfoNCE loss, which requires significant hardware resources. Due to the dependency of large batch, dense retriever has bottleneck of application and research. Recently, memory reduction methods have been broadly adopted to resolve the hardware bottleneck by decomposing forward and backward or using a memory bank. However, current methods still suffer from slow and unstable training. To address these issues, we propose Contrastive Accumulation (ContAccum), a stable and efficient memory reduction method for dense retriever trains that uses a dual memory bank structure to leverage previously generated query and passage representations. Experiments on widely used five information retrieval datasets indicate that ContAccum can surpass not only existing memory reduction methods but also high-resource scenario. Moreover, theoretical analysis and experimental results confirm that ContAccum provides more stable dual-encoder training than current memory bank utilization methods.
tags: []
# Display this page in the Featured widget?
featured: false

# links:
url_pdf: 'https://arxiv.org/abs/2406.12356v1'
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''
---