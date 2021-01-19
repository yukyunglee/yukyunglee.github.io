---
layout: post
title: Fisher-Information-Matrix
featured-img: sleek
---
Overcoming catastrophic forgetting in neural networks (EWC) 논문을 읽다가 Fisher information matrix를 활용한 loss function을 접하게 되었다. 하지만 FMI의 의미를 명확하게 설명해주는 자료를 찾지 못한것 같아 정리해보고자 한다. (증명은 많은데 정확히 의미하는 바를 알 수 없었다 😂)

검색해보면 위키피디아에 설명이 쭉 나오는데 아래와 같은 정보로는 정확한 의미를 이해하기엔 무리였다.

> 확률변수 ![X](https://wikimedia.org/api/rest_v1/media/math/render/svg/68baa052181f707c662844a465bfeeb135e82bab)가 미지의 매개변수 ![\theta ](https://wikimedia.org/api/rest_v1/media/math/render/svg/6e5ab2664b422d53eb0c7df3b87e1360d75ad9af)로 주어지는 분포를 따른다고 하자. 그렇다면, 관측값 ![{\displaystyle X=x}](https://wikimedia.org/api/rest_v1/media/math/render/svg/0661396d873679039ffe8e908a39f02402d4912d)으로부터 주어지는, ![\theta ](https://wikimedia.org/api/rest_v1/media/math/render/svg/6e5ab2664b422d53eb0c7df3b87e1360d75ad9af)에 대한 **피셔 정보** ![{\displaystyle {\mathcal {I}}(\theta )}](https://wikimedia.org/api/rest_v1/media/math/render/svg/93d0e554bc0fb296dac5ded2a7be914f4398543e)는 다음과 같다.

<img src="../_img/posts/Fisher.png" style="zoom:50%;" />

가장 큰 도움을 받은 자료는 9년전 업로드 된 [유튜브링크](https://www.youtube.com/watch?v=m62I5_ow3O8&ab_channel=jonathanpober) 였고, likelihood function으로 부터 차근차근 의미를 살펴본다.

* Fisher information이 크다

  = more curved

  = more peaky 

  = more constraining data (in particular parameter)



[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

