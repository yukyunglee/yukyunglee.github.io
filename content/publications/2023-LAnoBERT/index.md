---
title: "LAnoBERT: System log anomaly detection based on BERT masked language model"
authors:
- admin
- Jina Kim
- Pilsung Kang
author_notes:
- ''
- ''
- ''
date: "2023-10-01T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2023-10-01T00:00:00Z"

# Publication type.
publication_types: ["article-journal"]

# Publication name and optional abbreviated publication name.
publication: "Applied Soft Computing (IF = 8.7)"
abstract: The system log generated in a computer system refers to large-scale data that are collected simultaneously and used as the basic data for determining errors, intrusion and abnormal behaviors. The aim of system log anomaly detection is to promptly identify anomalies while minimizing human intervention, which is a critical problem in the industry. Previous studies performed anomaly detection through algorithms after converting various forms of log data into a standardized template using a parser. Particularly, a template corresponding to a specific event should be defined in advance for all the log data using which the information within the log key may get lost. In this study, we propose LAnoBERT, a parser free system log anomaly detection method that uses the BERT model, exhibiting excellent natural language processing performance. The proposed method, LAnoBERT, learns the model through masked language modeling, which is a BERT-based pre-training method, and proceeds with unsupervised learning-based anomaly detection using the masked language modeling loss function per log key during the test process. In addition, we also propose an efficient inference process to establish a practically applicable pipeline to the actual system. Experiments on three well-known log datasets, i.e., HDFS, BGL, and Thunderbird, show that not only did LAnoBERT yield a higher anomaly detection performance compared to unsupervised learning-based benchmark models, but also it resulted in a comparable performance with supervised learning-based benchmark models.
tags: []
# Display this page in the Featured widget?
featured: false

# links:

url_pdf: 'https://www.sciencedirect.com/science/article/pii/S156849462300707X'
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''
---