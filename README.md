# HSG: Hyperbolic Scene Graph
This is the official repository for the paper:
> **HSG: Hyperbolic Scene Graph**
>
> [Liyang Wang](https://github.com/lw1120)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, and [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.

## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{wang2025hsg,
  title={HSG: Hyperbolic Scene Graph},
  author={Wang, Liyang and Zhang, Zeyu and Tang, Hao},
  journal={European Conference on Computer Vision},
  year={2025},
  note={Under Review}
}
```

## 🏃 Intro HSG
HSG is a hyperbolic **representation learning framework** that models scene graphs in non-Euclidean space to better capture hierarchical relationships for structured 3D scene understanding.

Scene graph representations enable structured visual understanding by modeling objects and their relationships, and have been widely used for multiview and 3D scene reasoning. Existing methods such as MSG learn scene graph embeddings in Euclidean space using contrastive learning and attention-based association. However, Euclidean geometry does not explicitly capture hierarchical entailment relationships between places and objects, limiting the structural consistency of learned representations. To address this, we propose Hyperbolic Scene Graph (HSG), which learns scene graph embeddings in hyperbolic space where hierarchical relationships are naturally encoded through geometric distance. Our results show that HSG improves hierarchical structure quality while maintaining strong retrieval performance. The largest gains are observed in graph-level metrics: HSG achieves a PP IoU of **33.17** and the highest Graph IoU of **33.51**, outperforming the best AoMSG variant (25.37) by **8.14**, highlighting the effectiveness of hyperbolic representation learning for scene graph modeling.

![image](./model.png)

## ⚡ Quick Start
### Environment Setup




## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIGeeksGroup/HSG&type=Date)](https://www.star-history.com/#AIGeeksGroup/HSG&Date)

## 😘 Acknowledgement
We thank the authors of XXX for their open-source code.
