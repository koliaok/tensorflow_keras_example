Word Embedding Example  
=============
* text8 이란 영어 데이터를 바탕으로 word representation 을 배우는 word2vec example 작성
* 기존과 다르게 Keras Version을 만들려고 했으나, Keras로 잘 정리된 Tensorflow 공식 사이트에 모든 예제가 있습니다. 
* https://www.tensorflow.org/tutorials
* Keras 구현시 위 주소를 참고
* Word Representation:
  * 기계가 이해하기 위해서 단어를 어떻게 표현할 것인가? 
  * 기존: One Hot, Bow, tf-idf ... 
    * 진짜 단어의 의미를 잘 표현 해주는게 맞아? 
  * 단어의 의미를 Vector로 잘 표현 해보겠어 -> Word2vec
    * 1) BCOW, 2) Skip-gram
    * 위 두 방식의 가장 Basic Idea는 특정 단어 주위의 단어와 어떻게 연관이 되어있는지 Vector로 표현하자는 것
  * 한국어 적용시 문제 
    * 한국어 Data는 영어처럼 딱 들어선 문법을 사용하지 않는다. -> 단어의 사이의 의미 파악 어려움
    * NLP 어려움: 
        * 조사, 객체명인식(모든 언어 문제임) -> 이 부분이 참어려운데, 잘 만들어진 형태소분석기, 전처리기등이 그래서 필요
            * Mecabe, KKMA, Kakao 카이, ... 형태소 분석기는 많지만, Text의 도메인에 따라서 형태소와 전처리를 Customize 할 경우가 많음
  * word2vec의 문제
    * Target 단어를 학습하기 위해서 주변 단어를 몇개까지 볼 것인지 설정하는 window size 자체 문제 
    * 단어 자체의 뜻을 파악하지 못함. 
        * 주변 단어를 보고 Word Representation을 학습하는데, 같은 위치에 같은 단어가 다른 의미로 사용된다면?
        * Ex) 'bank' of america, 'bank' of river 
        * 따라서 주변 단어가 아니라 문장 전체에서 특정 단어가 어떤 의미를 가지는 학습하는 것이 필요 
            * BERT방식
* 위에서 나의 지식으로 단점만 말했는데, 이 방식이 논문으로 나오고 그 당시는 엄청 Hot한 기술이였고, 지금도 이를 보안하는 기법이 많이 사용되고 있다.
* 분명 NLP에서는 Word2vec이 Base지식임을 충분히 강조하고 싶고, 잘 설명된 사이트가 많다.(워낙 Hot했으니...)
* 이 블로그가(https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/) 잘 설명이 되었다고 생각되며, 나 또한 참고를 많이한 기억이 난다
* 코드 예제는 Jupyter에 Tensorflow 2.0으로 만듬