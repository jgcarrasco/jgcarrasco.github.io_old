---
title: "Diffusion Models"
date: 2020-09-15T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["diffusion", "generative"]
author: "Jorge García Carrasco"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
math: true
hidemeta: false
comments: false
description: "Learning about diffusion models"
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/jgcarrasco/jgcarrasco.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Resources

[1] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015, June). Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (pp. 2256-2265). PMLR.

[2] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33, 6840-6851. 

- The annotated diffusion model (https://huggingface.co/blog/annotated-diffusion)

- What are diffusion models? (https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)



## Diffusion Models

Diffusion models are generative models which could be an interesting alternative to GANs, as we will see in this post. In fact, this kind of models are used on DALLE-2 to get astonishing results (https://openai.com/dall-e-2/) (https://www.reddit.com/r/dalle2/)

The first diffusion model was presented in 2015 by Sohl-Dickstein [1]. The main idea, which was inspired by the statistical physics literature, consisted on **gradually adding noise** to the original sample until we get an easy, tractable distribution (typically a Gaussian), and then **learning the reverse process** in order to recover the original distribution. Therefore, If we are able to effectively model the reverse process with a DL model, we can sample from a simple, Gaussian distribution and iteratively denoise it to get a sample from the original distribution. 

- Brief description about diffusion models: forward and reverse process, main idea

- We will be focused mainly on the Ho paper

- How to model with NN, the loss is just the KL divergence between gaussians

- The reverse process is tractable when conditioned on x_o

![name](/images/DPM.PNG)