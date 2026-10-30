# Dataset source and attribution

## WikiText-103 raw v1

The reproducible training script downloads the original WikiText-103 raw archive from Salesforce's published location.

- Dataset description: https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/
- Archive: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
- Archive SHA-256: 91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794
- Paper: Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher (2016), Pointer Sentinel Mixture Models, https://arxiv.org/abs/1609.07843

The dataset contains over 100 million tokens from verified Wikipedia articles. Dataset-card metadata lists CC BY-SA 3.0 and GFDL identifiers; versioned metadata has also described CC BY-SA 4.0. Because licence metadata differs between published versions, review the exact archive's terms and attribution requirements before using or redistributing model weights, checkpoints or other derivatives. The downloader writes a separate attribution file next to the extracted data.

The preparation script creates distinct training, validation and test text files. BPE fitting uses a distributed sample from the training file only. Model updates use the training split; validation is for monitoring; the test split is held out for the final evaluation command.

WikiText-103 is English encyclopaedic prose, not a balanced instruction dataset. It is useful for a larger next-token pretraining experiment, but does not itself teach reliable assistant behaviour or guarantee broad capability.

## Reproducibility and storage

The source archive is approximately 192 MB; extracted data consumes substantially more disk space. The script verifies SHA-256 before extraction and checks archive paths against path traversal. Data and checkpoints are excluded from Git by .gitignore.
