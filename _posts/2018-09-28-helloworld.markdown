---
layout:     post
title:      "Welcome to  Blog"
subtitle:   " \"Hello World, Hello Blog\""
date:       2018-09-28 12:00:00
author:     "FxZ"
header-img: "img/post-bg-2018-001.jpg"
featured-tags: Life
tags:
    - 生活
---

> “Yeah It's on. ”


## 前言

FxZ 的 Blog 就这么开通了。

[跳过废话，直接看技术实现 ](#build)



2018 年，FxZ 打算找个地方来记录下生活，好好写点东西，以后也算可以有些实质性的回忆。

作为一个程序员， Blog 这种轮子要是挂在大众博客程序上就太没意思了。一是觉得大部分 Blog 服务都太丑，二是觉得不能随便定制不好玩。之前因为技术面太窄就没有自己搭建，结果就一直连
个写 Blog 的地儿都没有。
然而我又是个爱折腾的人，总是能在自己做了点东西之后，得到很大的满足感。

所以一不做二不休，查技术文档搞一个吧！


<p id = "build"></p>
---

## 正文

接下来说说搭建这个博客的技术细节。  

偶然发现[GitHub Pages](https://pages.github.com/) + [Jekyll](http://jekyllrb.com/) 快速 Building Blog 的技术方案

其优点非常明显：

* **Markdown** 带来的优雅写作体验
* 非常熟悉的 Git workflow ，**Git Commit 即 Blog Post**
* 利用 GitHub Pages 的域名和免费无限空间，不用自己折腾主机
	* 如果需要自定义域名，也只需要简单改改 DNS 加个 CNAME 就好了
* Jekyll 的自定制非常容易，基本就是个模版引擎

[^_^]: # (
本来觉得最大的缺点可能是 GitHub 在国内访问起来太慢，所以第二天一起床就到 GitCafe(Chinese GitHub Copy) 迁移了一个[镜像](http://huxpro.gitcafe.io)出来，结果还是巨慢。

哥哥可是个前端好嘛！ 果断开 Chrome DevTool 查了下网络请求，原来是 **pending 在了 Google Fonts** 上，页面渲染一直被阻塞到请求超时为止，难怪这么慢。  
忍痛割爱，只好把 Web Fonts 去了（反正超时看到的也只能是 fallback ），果然一下就正常了，而且 GitHub 和 GitCafe 对比并没有感受到明显的速度差异，虽然 github 的 ping 值明显要高一些，达到了 300ms，于是用 DNSPOD 优化了一下速度。)



---

配置的过程中也没遇到什么坑，顺便学习了 Git 的流程，在bash里与GitHub协同也很顺手

大的 Jekyll 主题上直接 fork 了 Clean Blog
[^_^]: # (这个主题也相当有名，就不多赘述了。唯一的缺点大概就是没有标签支持，于是HuX给它补上了。)

本地调试环境需要 `gem install jekyll`。

Theme 的 CSS 是基于 Bootstrap 定制的，看得不爽的地方直接在 Less 里改就好了。

最后就进入了耗时反而最长的**做图、写字**阶段，也算是进入了**写博客**的正轨，因为是类似 Hack Day 的方式去搭这个站的，所以折腾折腾着大半夜就过去了。

第二天考虑中文字体的渲染，fork 了 [Type is Beautiful](http://www.typeisbeautiful.com/) 的 `font` CSS，调整了字号，适配了 Win 的渣渲染，中英文混排效果好多了。


## 后记
我在萌生了想要找个地方写点东西的想法以后，偶然在知乎上看到了HuX的Blog，觉得
>嗯，这大概就是我想找的。

引用HuX原话

>在经历 v1.0 - v1.5 的蜕变后，这个博客主题愈发完整，不但增加了诸多 UI 层的优化（opinionated）；在代码层面，更加丰富的配置项也使得这个主题拥有了更好的灵活性与可拓展性。而作为一个开源项目，我也积极的为其完善文档与解决 issue。

如果你恰好逛到了这里，希望你也能喜欢这个博客主题。

—— FxZ 后记于 2018.11
