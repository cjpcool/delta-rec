This package contains modified, dependency-minimized components from the following projects. Existing third-party copyright statements remain applicable; they do not identify the authors of DeltaRec.

- Meta generative-recommenders: HSTU/DLRMv3 components and operators, copyright Meta Platforms, Inc. and affiliates; Apache 2.0, included in `licenses/Apache-2.0.txt`. https://github.com/meta-recsys/generative-recommenders
- FuXi-Linear: model components, copyright 2025-2026 USTC-StarTeam; per-file Apache 2.0 notices, included license above. https://github.com/USTC-StarTeam/fuxi-linear
- RecBole: residual/normalization/feed-forward components, copyright (c) 2020 RUCAIBox; MIT, included in `licenses/RecBole-MIT.txt`. https://github.com/RUCAIBox/RecBole
- LinRec attention and BlossomRec initialization/feed-forward adaptations: https://github.com/Applied-Machine-Learning-Lab/LinRec and https://github.com/Applied-Machine-Learning-Lab/WWW2026_BlossomRec. No explicit redistribution license was located for these additions. Permission remains unresolved; this bundle is not cleared for public/reviewer redistribution until that is resolved. RecBole's MIT license alone is not asserted to license these additions.

Modifications extract components, change import namespaces and file locations, replace environment-specific loading with local tensor checkpoints, and preserve the selected DeltaRec training and evaluation operations. No independent baseline training/evaluation pipelines are included.

KuaiRand-1K data are provided under Creative Commons Attribution-ShareAlike 4.0 (the archive license is included as `licenses/KuaiRand-CC-BY-SA-4.0.txt`). Dataset: Chongming Gao et al., *KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos*, CIKM 2022, https://doi.org/10.1145/3511808.3557624 . Derived numeric mappings/groupings are modified preprocessing outputs; the original dataset is available at https://zenodo.org/records/10439422 .
