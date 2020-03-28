Convolution Mnist Example
============
이미지 손 글씨 숫자 데이터 예제 
기본 28x28 이미지 데이터
convolution 기법을 이용한 이미지 분류 학습 예제

* 0~9까지 숫자 데이터를 입력을 받아서 숫자를 예측하는 문제 

# Convolution 학습 
1. Convolution Layer 
    1. 이미지 전체 패턴을 Kernel Map으로 이미지 Feature 특징을 Convolution
    연산으로 학습. 
        * Convolution 연산 ex)
            * 개와 고양이를 구분할 때 부분적인 이미지 패턴(귀, 눈, 코, 꼬리, 등)을 파악 
            * Convolution 연산이 이런 특징들을 학습함
2. Pooling연산
    1. Conv연산으로 인한 Overfitting을 피하고, 학습된 Feature의 특징점을 뽑아내는 역할
    
* CNN network 참고하기 좋은 자료
    * [CONVNET (CNN) 이해하기](https://de-novo.org/2018/05/27/convnet-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0/) 참조
    * [TAEWAN.KIM 블로그](http://taewan.kim/post/cnn/) 참조