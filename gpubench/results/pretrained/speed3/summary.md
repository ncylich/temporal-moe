| arm | setting | texts | tok/s | range over texts | ms per token | loads per decoded token-layer | peak VRAM MiB |
|---|---|---|---|---|---|---|---|
| base | vram | 5 | 655.8 | 655.3 to 656.4 | 1.525 | -- | 1438 |
| ours_stream | k2-x0 | 5 | 580.2 | 576.3 to 585.9 | 1.724 | 1.000 | 918 |
| ours_stream | k2-x1 | 5 | 585.1 | 584.3 to 586.0 | 1.709 | 1.000 | 918 |
| ours_stream | s0x0 | 5 | 602.5 | 598.6 to 604.1 | 1.660 | 0.999 | 918 |
| ours_stream | s0x1 | 5 | 599.0 | 595.5 to 604.4 | 1.670 | 0.999 | 918 |
| ours_stream | s1x0 | 5 | 612.4 | 609.1 to 616.4 | 1.633 | 0.999 | 918 |
| ours_stream | s1x1 | 5 | 614.7 | 610.2 to 618.1 | 1.627 | 0.999 | 918 |
| ours_stream | s2x0 | 5 | 611.9 | 606.4 to 617.4 | 1.634 | 0.999 | 918 |
| ours_stream | s2x1 | 5 | 614.2 | 608.8 to 618.5 | 1.628 | 0.999 | 918 |
| ours_vram | k2 | 5 | 673.0 | 671.8 to 674.1 | 1.486 | 1.000 | 1438 |
| ours_vram | vram | 5 | 699.1 | 696.9 to 700.2 | 1.430 | 0.999 | 1438 |
| skliar_stream | fullsel-x0 | 5 | 551.1 | 543.7 to 556.8 | 1.815 | 1.055 | 918 |
| skliar_stream | fullsel-x1 | 5 | 551.3 | 547.1 to 556.8 | 1.814 | 1.055 | 918 |
| skliar_stream | s0x0 | 5 | 571.4 | 567.1 to 578.0 | 1.750 | 1.059 | 918 |
| skliar_stream | s0x1 | 5 | 572.4 | 560.5 to 580.4 | 1.747 | 1.059 | 918 |
| skliar_stream | s1x0 | 5 | 581.1 | 573.1 to 587.7 | 1.721 | 1.059 | 918 |
| skliar_stream | s1x1 | 5 | 577.8 | 575.5 to 578.7 | 1.731 | 1.059 | 918 |
| skliar_stream | s2x0 | 5 | 580.8 | 578.9 to 584.2 | 1.722 | 1.059 | 918 |
| skliar_stream | s2x1 | 5 | 582.0 | 574.8 to 588.4 | 1.718 | 1.059 | 918 |
| skliar_vram | fullsel | 5 | 645.2 | 643.3 to 646.8 | 1.550 | 1.057 | 1438 |
| skliar_vram | vram | 5 | 673.1 | 670.8 to 674.8 | 1.486 | 1.054 | 1438 |
| trained_stream | s0x0 | 5 | 585.2 | 583.8 to 586.9 | 1.709 | 0.999 | 918 |
| trained_stream | s0x1 | 5 | 584.2 | 577.9 to 586.4 | 1.712 | 0.999 | 918 |
| trained_stream | s1x0 | 5 | 597.6 | 597.1 to 598.3 | 1.673 | 0.999 | 918 |
| trained_stream | s1x1 | 5 | 596.5 | 589.3 to 598.6 | 1.676 | 0.999 | 918 |
| trained_stream | s2x0 | 5 | 595.5 | 589.8 to 598.5 | 1.679 | 0.999 | 933 |
| trained_stream | s2x1 | 5 | 597.6 | 597.0 to 598.6 | 1.673 | 0.999 | 918 |
| trained_vram | vram | 5 | 677.3 | 676.8 to 678.1 | 1.476 | 0.999 | 1438 |

Each streamed method at its fastest setting:

| arm | fastest setting | tok/s | ms per token |
|---|---|---|---|
| ours_stream | s1x1 | 614.7 | 1.627 |
| skliar_stream | s2x1 | 582.0 | 1.718 |
| trained_stream | s1x0 | 597.6 | 1.673 |
