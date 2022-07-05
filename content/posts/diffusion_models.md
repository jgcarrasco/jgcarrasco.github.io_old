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
    URL: "https://github.com/jgcarrasco/jgcarrasco.github.io/tree/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

In this post, we will learn about a kind of generative models called diffusion models, which are a promising alternative to GANs, as they are an important piece of the impressive DALLE-2 model. As we will see, the main idea is simple and intuitive, but it has some details that have to be taken into account. The aim of this post is to act as a **simple introduction to understanding and implementing a diffusion model**, as well as listing the resources that I found useful when learning myself. 

## Resources

[1] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015, June). Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (pp. 2256-2265). PMLR.

[2] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33, 6840-6851. 

- The annotated diffusion model (https://huggingface.co/blog/annotated-diffusion)

- What are diffusion models? (https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)



## Brief introduction to Diffusion Models

Diffusion models are generative models which could be an interesting alternative to GANs, as we will see in this post. In fact, this kind of models are used on DALLE-2 to get astonishing results (https://openai.com/dall-e-2/) (https://www.reddit.com/r/dalle2/)

The first diffusion model was presented in 2015 by Sohl-Dickstein [1]. The main idea, which was inspired by the statistical physics literature, consisted on **gradually adding noise** to the original sample until we get an easy, tractable distribution (typically a Gaussian), and then **learning the reverse process** in order to recover the original distribution. Therefore, If we are able to effectively model the reverse process with a DL model, we can sample from a simple, Gaussian distribution and iteratively denoise it to get a sample from the original distribution.

Formally, the **forward diffusion process** is defined by the successive application of some distribution, typically a Gaussian:

$$q(x_t \mid x_{t-1})=\mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbb{I})$$

If we apply this process long enough and with a well behaved variance schedule $\beta_1, ..., \beta_T$, we will obtain an isotropic Gaussian distribution (Fig. 1).

Now, if we could be able to obtain the **reverse process**, $q(x_{t-1}\mid x_t)$, we could sample from a Gaussian distribution and iteratively apply $q$ until we get a generated sample from the original distribution. Seems easy, right? Well, it turns out that the reverse process depends on the original data distribution, which is what we're actually trying to model, so obtaining this is **untractable**. This is where Deep Learning comes into play, by trying to approximate this reverse process by a function $p_{\theta}(x_{t-1}\mid x_t)$ represented by a neural network.

![name](/images/ddpm.PNG)
*Figure 1: Diagram of the forward and reverse diffusion processes (Source: Ho et al. 2020 [1])*

## Going deeper: Denoising Diffusion Probabilistic Models

Now that the main idea of diffusion networks has been understood, it is time to study the mathematical details that we need to start implementing the model. We will be mainly based on the paper presented by Jonathan Ho et al. in 2020 [1], where they extend the idea by developing the so-called **Denoising Diffusion Probabilistic Models (DDPM)**.

As the forward process is a successive application of gaussians, we can expect that the reverse process can be also modelled in the same way,

$$p_\theta (x_{t-1}\mid x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

therefore, we just need to model the mean and variance of a gaussian distribution. In fact, Ho et al. proved that fixing the variance to $\Sigma_\theta = \sigma_t^2 \mathbb{I}$ gave good results, as we will see later.


We want to **minimize the negative log-likelihood** of $p_\theta$. We are not able to directly optimize this, but it can be bounded similar to an autoencoder, with the **variational lower bound (VLB)**:

$$\mathbb{E}\left[-\log p_\theta(x_0) \right] \leq \mathbb{E} \left[ -\log p(x_T) - \sum_{t \geq 1}\log \dfrac{p_\theta (x_{t-1} \mid x_t)}{q(x_t | x_{t-1})}\right] = L$$

Now the authors propose several tricks and reparameterizations that greatly simplifies the training process. We will list and explain all of them, but refer to [1] or [this post of Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) for the mathematical derivation:

-------------

**1.** We can sample $x_t$ in closed form, instead of iteratively applying the forward diffusion step,

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t) \mathbb{I})$$

where $\alpha_t = 1 - \beta_t$, $\bar{\alpha_t} = \prod_{s=1}^t \alpha_s$.

----------

**2.** The reverse process $q$ is tractable when conditioned on $x_0$:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbb{I})$$

where $\tilde{\mu} _t (x _t, x _0) = \dfrac{\sqrt{\bar{\alpha} _{t-1}}\beta _t}{1 - \bar{\alpha _t}}x _0 + \dfrac{\sqrt{\alpha _t}(1 - \bar{\alpha _{t-1}})}{1 - \bar{\alpha _ t}}x_t$ and $\tilde{\beta} _ t = \dfrac{1 - \bar{\alpha} _{t-1}}{1 - \bar{\alpha} _t} \beta _t$ 

-----------

**3.** We can rewrite the loss as follows:

$$L = \mathbb{E} _q \left[\underbrace{-\log p _\theta(x_0 \mid x_1)} _{L_0} + \sum _{t=2}^T \underbrace{D _{KL}(q(x _{t-1} \mid x _t, x _0) \parallel p _\theta (x _{t-1} \mid x _t))} _{L _{t-1}} + \underbrace{D _{KL}(q(x_T \mid x_0) \parallel p _\theta (x_T))} _{L_T} \right]$$

Which, except for the term $L_0$, is composed by the KL divergences between two Gaussians, which can be [evaluated in closed form](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions)

--------------

**4.** If we set $\beta_t$ to an arbitrary chosen schedule (i.e. we don't treat it as a trainable parameter) and set $\Sigma_\theta = \sigma_t^2 \mathbb{I}$, where $\sigma_t$ is a function of $\beta_t$, then we can discard the $L_T$ term, the $L_0$ term can also be discarded, as $p _\theta (x _0 \mid x _1)$ is replaced by a discrete decoder ([1], Section 3.3), and the $L _{t-1}$ terms become:

$$L_{t-1} = \mathbb{E} \left[ \dfrac{1}{2 \sigma_t^2} \lVert \tilde{\mu}_t (x_t, x_0) - \mu _\theta (x_t, t) \rVert^2 \right]$$

therefore, we could use a neural network to model $\mu _\theta$ and train it to approximate $\tilde{\mu}_t$, but they make one more parameterization. According to **1**, we can write:

$$x_t(x_0, \epsilon) = \sqrt{\bar{\alpha} _t} x_0 + \sqrt{1 - \bar{\alpha} _t \epsilon}$$

using this and **2**, we get the following loss term:

$$\mathbb{E} _{x_0, \epsilon} \left[ \dfrac{\beta _t^2}{2 \sigma _t^2 \alpha _t (1 - \bar{\alpha} _t)} \lVert \epsilon - \epsilon _\theta (\sqrt{\bar{\alpha} _t} x _0 + \sqrt{1 - \bar{\alpha} _t} \epsilon, t) \rVert^2\right]$$

i.e., we can use $\mu _\theta$ to model $\tilde{\mu}$, or **we can use $\epsilon _\theta (x_t, t)$ to model the noise $\epsilon$**. The final important change that they make to the loss function is to **discard the weighting term**, as they found that it gave better experimental results.

---

Finally, we have worked out all the details needed to start implementing a DDPM! Below we show a brief description of the training and sampling processes, as well as its pseudocode:

**Training:**
- Sample an image from the original distribution, $x_0$, a random timestep $t \in \[1, ..., T \]$ and noise $\epsilon \sim \mathcal{N}(\mathbb{0}, \mathbb{I})$
- Train a gradient descent step on the corresponding $L_t$ term of the total loss (Fig. 2).
- At each step of the training process, one term of the loss chosen at random will be minimized. Repeat until it converges.

**Sampling:**
- Sample a noise image $x_T \sim \mathcal{N}(\mathbb{0}, \mathbb{I})$ from the "diffused" distribution.
- Now, we are going to perform the reverse process by iteratively using the learned noise function to slowly denoise the sample:
  - If $t > 1$, then sample noise $z$ from a gaussian distribution of mean 0 and variance 1, else $z = 0$ (in the last step of the reverse process, we will not apply noise).
  - Perform a step on the reverse process $x _{t-1} = \dfrac{1}{\sqrt{\alpha _t}} \left( x _t - \dfrac{1 - \alpha _t}{\sqrt{1 - \bar{\alpha}} _t} \epsilon _\theta (x _t, t) \right)$
- Return $x_0$

![algorithm](/images/algorithm.png)
*Figure 2: Algorithms for training and sampling from a DDPM. Source [2]*

## Let's start coding: Testing on a simple distribution
