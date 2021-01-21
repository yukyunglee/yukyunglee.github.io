---
layout: post
title: Fisher Information Matrix
summary: Fisher Information Matrix 정리 😎
featured-img: Fisher_head
categories: Statistics
use_math: true
---
Overcoming catastrophic forgetting in neural networks (EWC) 논문을 읽다가 Fisher information matrix를 활용한 loss function을 접하게 되었다. 하지만 FMI의 의미를 명확하게 설명해주는 자료를 찾지 못한것 같아 정리해보고자 한다. (증명은 많은데 정확히 의미하는 바를 알 수 없었다 😂)

검색해보면 위키피디아에 설명이 쭉 나오는데 아래와 같은 정보로는 정확한 의미를 이해하기엔 무리였다. 

> 확률변수 ![X](https://wikimedia.org/api/rest_v1/media/math/render/svg/68baa052181f707c662844a465bfeeb135e82bab)가 미지의 매개변수 ![\theta ](https://wikimedia.org/api/rest_v1/media/math/render/svg/6e5ab2664b422d53eb0c7df3b87e1360d75ad9af)로 주어지는 분포를 따른다고 하자. 그렇다면, 관측값 ![{\displaystyle X=x}](https://wikimedia.org/api/rest_v1/media/math/render/svg/0661396d873679039ffe8e908a39f02402d4912d)으로부터 주어지는, ![\theta ](https://wikimedia.org/api/rest_v1/media/math/render/svg/6e5ab2664b422d53eb0c7df3b87e1360d75ad9af)에 대한 **피셔 정보** ![{\displaystyle {\mathcal {I}}(\theta )}](https://wikimedia.org/api/rest_v1/media/math/render/svg/93d0e554bc0fb296dac5ded2a7be914f4398543e)는 다음과 같다.

<div style="text-align:center"><img src="../assets/img/posts/Fisher.jpg" style="zoom:50%;" /></div>



가장 큰 도움을 받은 자료는 9년전 업로드 된 [유튜브링크](https://www.youtube.com/watch?v=m62I5_ow3O8&ab_channel=jonathanpober) 였고, likelihood function으로 부터 차근차근 의미를 살펴본다. likelihood function는 아래의 식으로 정의 할 수 있다. (여기서 $f$는 probability density function을 나타내는데 대부분 가우시안으로 가정한다. )


$$
\mathcal{L} = \mathcal{L}(\theta|X)=f(X|\theta) = f(x_1,x_2,...,x_n|\theta)
$$
영상에서는 $\mathcal{L} = P(data|thory)$ 로 바로 정의내린 후 가우시안을 가정해서 설명을 진행한다. 

요약하자면, Talyor expansion을 통해 likelihood function을 근사한 후 multi parameter 상황에서 Fisher information matrix를 유도하는 방향으로 진행된다. (유도과정은 [링크](https://stats.stackexchange.com/questions/174600/help-with-taylor-expansion-of-log-likelihood-function)에 잘 정리되어있다) Fisher information matrix 는 likelihood function을 두번 미분한 값으로 이해할 수 있기 때문에 'Curvature matrix'로도 불리며 likelihood function이 얼마나 커브 되어있는지 알 수 있다. 

결국 관측값으로 부터 주어지는 $\theta$에 대한  **Fisher information** 는 **'얼마나 커브 되었는가'** 라는 정보라고 결론 낼 수 있다. 그리고 아래와 같이 이해하면 조금 더 쉽다 !



* **Fisher information이 크다**

  = more curved

  = more peaky 

  = more constraining data (in particular parameter)



추가적으로 가우시안에서는 Fisher information matrix의 역수가 covariance matrix 와 같다 . Overcoming catastrophic forgetting in neural networks (EWC) 논문에서 fisher informaion matrix의 diagonal 값을 사용하여 loss term을 정의하기 때문에 논문에서 말하는 값이 각 파라미터의 분산의 역수(=정밀도,precision) 인것도 알 수 있었다. EWC는 loss term에 fisher information을 추가하여 regularization을 함으로서 catastropy forgetting을 줄이는 방향으로 학습을 진행하는 방법론이라는 결론을 내렸다.

딥러닝에서 Fisher information matrix는 큰 장점을 가지는데, hessian matrix와 같다고 봐도 되기 때문에 연산 효율성을 위해 Fisher information matrix로 근사해서 문제를 푼다고 한다. [링크](https://nzer0.github.io/Fisher-Info-and-NLL-Hessian.html)에서 관련개념을 자세히 정리해주시니 참고하면 좋을 것 같다.



[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

