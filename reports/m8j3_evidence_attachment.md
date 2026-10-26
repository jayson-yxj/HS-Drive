# Official evidence index

Prepared 2026-10-26. Official-source statements are inconsistent; this index does not resolve their scope. No dataset contents are attached.

| Evidence | Official source | Locally retained evidence |
|---|---|---|
| NeurIPS paper, p.25, I License | [Official PDF](https://papers.nips.cc/paper/2024/file/017761f94a1cd66d01c041aff85492c4-Paper-Datasets_and_Benchmarks_Track.pdf) | 8022848 bytes; SHA256 `9be21e65da597ffcc13c5102b00866cd3e3a150d1741235fe684e050e635356a`; local full copy, [excerpt/hash](m8j3_sources/paper_archive.json) |
| LICENSE change | [fd3009a16f551c6d36139c831a4a172f17b74419](https://github.com/Thinklab-SJTU/Bench2Drive/commit/fd3009a16f551c6d36139c831a4a172f17b74419) | [Official patch](m8j3_sources/license_commit.patch), SHA256 `ace80773db37b170c23d51cc0edea9952cd635754f77482ca23a630e360bd594` |
| README change | [8f9f1da103ea7153a17c5b1dd046548a9bcd5af2](https://github.com/Thinklab-SJTU/Bench2Drive/commit/8f9f1da103ea7153a17c5b1dd046548a9bcd5af2) | [Official patch](m8j3_sources/readme_commit.patch), SHA256 `e07ae7ea8349cb7d0002e0f9269d7dea56d934253b7b3d6bd66d3f578f2e5829` |
| HF data card | [Pinned README](https://huggingface.co/datasets/rethinklab/Bench2Drive/blob/ce56cfbc26edf7796586e26fd44644f0ac2f6bcc/README.md) | J2 [raw text](m8j2_sources/http_06.body), 2096 bytes; SHA256 `50b6d7dc1d8efa76b813b0b32b2cedccbf25461b363ce3b3312b019787de412f` |
| Unique target archive | [Official Mini list at 7ec25d1c9f7522d923ce5f3420986cef1cb2d956](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/docs/bench2drive_mini_10.json) | `DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz`; 280176944 bytes; SHA256 `b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2`; downloaded bytes independently match |

The target is the legacy Mini archive in the original repository, not v0.0.4 training/depth/occupancy data or Dev10/220 evaluation routes. Hash equality establishes byte identity with the official manifest; it does not establish rights beyond the applicable license.
