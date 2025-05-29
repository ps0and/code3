import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from streamlit_ace import st_ace
import matplotlib.pyplot as plt
import io, sys, textwrap

def show():
    # 1) 수열 예측 사전 지식 학습지 (Markdown 그대로)
    pre_knowledge = r'''
    ##### 1. **머신러닝으로 수열을 예측할 수 있을까?**

    - **수열 문제**: 2, 5, 8, 11 … 처럼 규칙을 찾아 다음 숫자를 맞추는 게임과 비슷함
    - **머신러닝의 장점**: 사람이 규칙을 일일이 수식으로 만들지 않아도, 예시(데이터)를 보여주면 기계가 스스로 패턴을 찾음
    - **응용 예시**:
        - 시험 점수 추세로 다음 시험 예상
        - 기온 변화로 내일 날씨 예측

    ##### 2. **머신러닝 학습 도구**
    - **데이터(수열)**: 학습할 숫자의 나열(수열)
    - **모델(머신러닝 알고리즘)**:
        - 선형 회귀(Linear Regression): 직선 추세선
        - 다항 회귀(Polynomial Regression) 곡선 추세선
    - **코딩 도구**:
        - Python
        - NumPy (숫자 계산)
        - scikit-learn (머신러닝 라이브러리)

    ##### 3. **핵심 개념: 어떻게 동작할까?**
    - **숫자에 ‘위치’를 붙여요**
        - 수열이 `[2, 5, 8, 11]`일 때
        - 위치 1 → 값 2
        - 위치 2 → 값 5
        - 위치 3 → 값 8
    - **모델에게 알려줘요(학습)**
        - 머신러닝 모델에 위치(입력값)와 값(출력값)을 함께 보여주면, 패턴을 스스로 학습합니다.
    - **다음 위치 예측하기**
        - 예: 5번째 숫자를 알고 싶다면 `5`를 모델에 넣으면, 예측값을 반환합니다.

    ##### 3. 회귀 분석(Regression) 개요

    - **회귀 분석이란?**
      - **수(number)** 를 예측하기 위한 통계적·머신러닝 기법.

    - **선형 회귀(Linear Regression)**
      - 데이터를 가장 잘 설명하는 직선 방정식을 찾음.
      - 예측값: $\hat y = w_0 + w_1 x$

      - $w_1$: 기울기(slope), $w_0$: 절편(intercept)
      - **손실 함수(Loss)**: $\displaystyle \mathrm{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat y_i)^2$

    - **다항 회귀(Polynomial Regression)**
      - 입력 $x$를 $[x, x^2, …, x^d]$로 확장 후 **선형 회귀** 적용
      - 예측값:
    $$
    \hat y ​= w_0​+w_1​x+w_2​x^2+⋯+w_n​x^n
    $$

    ##### 4. **실습 흐름**

    1. 수열 입력: `2, 5, 8, 11`
    2. 모델 선택: 선형 / 다항
    3. 학습 실행: `model.fit(X, y)`
    4. 예측 실행: `model.predict([[5]])`
    5. 결과 확인: 예측값과 실제값 비교

    > **Tip:** 앱의 슬라이더와 라디오 버튼으로 입력을 바꿔 가며, 예측 결과 변화를 살펴보세요!

    ##### 5. **마무리**

    - 머신러닝은 **데이터를 학습하는 도구**입니다.
    - 간단한 수열 예측부터 시작해, 점차 복잡한 데이터(날씨, 주가, 그림·소리 등)에도 적용할 수 있습니다.
    - 오늘 배운 **‘위치 → 값’** 개념과 **선형·다항 회귀**를 바탕으로, 직접 코드를 수정·확장해 보세요!
    '''   

    # 2) 수열 예측 머신러닝 교육 자료 (expander)
    ml_material = '''
    ##### 📖**학습 목표**

    - 주어진 파이썬 코드를 읽고, 머신러닝으로 수열을 예측하는 과정을 이해한다.
    - `NumPy`, `scikit-learn`의 주요 함수와 메서드를 활용해 입출력 데이터 형태를 변환하는 법을 익힌다.
    - 선형 회귀와 다항 회귀에서 각각 어떤 연산이 일어나는지 개념적으로 파악한다.

    ---
    ##### 📖**코드 설명**
    🔖 **선형 회귀 모델**

    ```python
     1) 필요한 라이브러리 불러오기
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
    ``` 
    - **`numpy`**: 배열 생성·연산을 쉽게 도와주는 패키지
    - **`LinearRegression`**: 직선 형태 모델로 학습·예측
    - **`PolynomialFeatures`**: 입력값을 다항식 항 $(x, x^2, x^3,...)$으로 확장하는 도구

    ```python 
    2) 학습용 수열과 예측할 항 번호
    seq = [2, 5, 8, 11]      # 학습할 수열 데이터
    n   = 5                  # 예측할 항의 번호 (5번째 항)
    ```
    - **`seq`**: 예시로 사용할 수열, 예) `[2, 5, 8, 11]`
    - **`n`**: 우리가 알고 싶은 "다음" 혹은 "특정" 항의 번호, 예) 5
    ```python
    3) 입력 X와 출력 y 준비
    X   = np.arange(1, len(seq)+1).reshape(-1, 1)
    # 위치 정보를 2차원 배열(열 벡터)로 변환
    y   = np.array(seq)       # 수열 값을 1차원 배열로 준비
    ```
    - `np.arange(1, len(seq)+1)` → `[1, 2, 3, 4]`: 수열 위치 생성
    - `.reshape(-1, 1)` → `[[1], [2], [3], [4]]`: 2차원 열 벡터 형태로 변환
    - `[2, 5, 8, 11]` → 1차원 배열
    - **1차원**은 ‘단일 출력’을 위한 자연스러운 형태
    ```python
     4) 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    ```
    - `fit(X, y)` → `X`와 `y`를 보고 **최적의 기울기와 절편**을 계산
    - 학습된 직선은  $\hat y= w_1 x + w_0 $ 형태
    ```python
    5) 예측 및 결과 출력
    pred = model.predict([[n]])[0]
    print(pred)  # n번째 예측값
    ```
    - `[[n]]` → shape `(1,1)`인 2차원 입력
    - `predict` 결과는 1차원 배열(`array([값])`) → `[0]`으로 스칼라 값을 추출
    - 예측값을 화면에 출력

    ---

    **🔖다항 회귀 모델 확장**
    ```python
    1) 다항 변환기 사용
    poly  = PolynomialFeatures(degree=3, include_bias=False)
    ```
    - `degree=3` → $[x, x^2, x^3]$ 특성 생성
    - `include_bias=False` → 상수항 $x^0=1$ 제외
    ```python
    2) 변환 후 학습
    Xp = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(Xp, y)
    ```
    - `Xp` 는 `X`를 $d$ 차항까지 확장한 2차원 배열
    - 동일한 `LinearRegression`으로 학습하되, 입력 특성이 늘어남

    ```python
    3) 다항 예측
    pred = model.predict(poly.transform([[n]]))[0]
    print(pred)
    ```
    - `poly.transform([[n]])` → $[n, n^2, n^3]$ 형태로 변환
    '''

    with st.expander("📚 **머신러닝의 이해**"):
        st.markdown(pre_knowledge)
    with st.expander("📝 **머신러닝 수열 예측 파이썬 코드의 이해**"):
        st.markdown(ml_material)
        
    st.divider() 
    st.markdown("### 🚀수열 예측")

    # 1) 모델 선택
    model = st.radio("모델을 선택하세요", ["LinearRegression", "PolynomialRegression"])
    degree = st.slider("다항 회귀 차수 선택", 2, 5, 2) if model=="PolynomialRegression" else None

    # 2) 수열 입력
    seq_input = st.text_input("수열을 입력하세요 (예: 2,5,8,11)", "2,5,8,11")

    # 3) 예측할 항 번호 입력
    term_idx = st.number_input("예측할 항 번호를 입력하세요",
                               min_value=1, value=len(seq_input.split(","))+1)

    # 4) 코드 템플릿 생성
    if model=="LinearRegression":
        raw = f"""
import numpy as np 
from sklearn.linear_model import LinearRegression

seq = [{seq_input}] #학습할 수열
n = {term_idx} #예측할 항의 번호

X = np.arange(1, len(seq)+1).reshape(-1,1) #입력값은 2차원 배열 변화
y = np.array(seq) #목표값은 1차원 배열

model = LinearRegression() #선형 회귀모델
model.fit(X, y) #머신러닝 학습

pred = model.predict([[n]])[0] #n번째 예측값
print(pred)
"""
    else:
        raw = f"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

seq = [{seq_input}] #학습할 수열
n = {term_idx} #예측할 항의 번호

X = np.arange(1, len(seq)+1).reshape(-1,1) #입력값은 2차원 배열 변화
y = np.array(seq) #목표값은 1차원 배열

poly = PolynomialFeatures(degree={degree}, include_bias=False) #n차 다항식으로 확장하는 변환기
Xp = poly.fit_transform(X) #입력값을 변환
model = LinearRegression() #선형 회귀모델
model.fit(Xp, y) #머신러닝 학습

pred = model.predict(poly.transform([[n]]))[0]
print(pred)
"""
    full_code = textwrap.dedent(raw)

    # 5) ACE 에디터: 항상 최신 full_code 반영
    signature = f"{model}|{seq_input}|{term_idx}|{degree}"
    st.markdown("#### 🖥️ 파이썬 코드 에디터 (수정 가능)")
    user_code = st_ace(
        value=full_code,
        language="python",
        theme="monokai",
        height=300,
        key=f"ace_{signature}"
    )

    # 6) 실행 및 시각화
    if st.button("▶️ 예측 실행 및 시각화"):
        buf = io.StringIO()
        # exec할 때 로컬 실행 공간을 dict로 만들어 seq, pred, n을 뽑아옵니다.
        exec_locals = {}
        try:
            sys.stdout = buf
            exec(user_code, {}, exec_locals)
        finally:
            sys.stdout = sys.__stdout__

        # 캡처된 프린트 결과
        output = buf.getvalue().strip()
        st.success(f"실행 결과: {output}")

        # 시각화
        seq = exec_locals.get("seq", [])
        pred = exec_locals.get("pred", None)
        n = exec_locals.get("n", None)

        if seq and (pred is not None) and (n is not None):
            fig, ax = plt.subplots()
            ax.plot(range(1, len(seq)+1), seq, marker="o", label="input sequence")
            ax.scatter(n, pred, color="red", label=f"{n}th predicted value")
            ax.set_title("Sequence and prediction results")
            ax.set_xlabel("number")
            ax.set_ylabel("value")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("`seq`, `pred`, `n` The value is not properly defined.")

# 실제 앱에서는 아래처럼 show()를 호출합니다.
if __name__ == "__main__":
    show()
