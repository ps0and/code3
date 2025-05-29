import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from streamlit_ace import st_ace
import io, sys, textwrap

# 코드 템플릿
RAW_CODE_TEMPLATE = """\
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 학생 입력값
seq = [{seq_input}]
n = {term_idx}

# 데이터 전처리
data = np.array(seq, dtype=float)
X, y = [], []
for i in range(len(data)-1):
    X.append([data[i]])
    y.append(data[i+1])
X = np.array(X).reshape(-1, 1, 1)
y = np.array(y)

# 3층 LSTM 모델 구성
model = Sequential([
    LSTM({units1}, return_sequences=True, input_shape=(1, 1)),
    LSTM({units2}, return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 모델 학습
model.fit(X, y, epochs={epochs}, verbose=0)

# 다음 항 예측
last_input = np.array([[[data[-1]]]])
pred = model.predict(last_input)[0, 0]

# 결과 출력
print()
print(f"✅ 예측된 {{n}}번째 항: {{pred:.2f}}")
"""

def show():
    pre_knowledge = '''
    ##### 1. **딥러닝이 뭐예요?**
    - **딥러닝**은 뇌 신경 세포처럼 생긴 층(layer)을 여러 개 쌓아 만든 컴퓨터 모델이에요.
    - **장점**: 사람처럼 데이터를 보고 스스로 규칙을 배우기 때문에, 복잡한 문제도 해결할 수 있어요.
    - **예시**: 사진 속 고양이 찾기, 음성으로 말 알아듣기, 주가나 날씨 예측 등

    ##### 2. **데이터 전처리 이해**
    - **수열(sequence)**: `[2, 5, 8, 11]` 같은 숫자 나열이에요.
    - **입력(X)과 정답(y)** 만들기  
    - `X[0] = 2` → `y[0] = 5`  
    - `X[1] = 5` → `y[1] = 8`  
    - **모양 바꾸기**: LSTM에 맞게 `(batch, timestep, feature)` 형태로 만들어요.
    - 여기서는 `(예시개수, 1, 1)`로 변환해요.

    ##### 3. **왜 여러 층(layer)을 쓸까요?**
    - **1층 LSTM**: 가까운 이웃 숫자 관계를 배워요.  
    - **2층 LSTM**: 더 깊은 패턴(장기 기억)을 잡아요.  
    - **Dense 층**: 마지막으로 다음 숫자를 하나만 뽑아내요.

    ##### 4. **어떻게 학습하고 예측해요?**
    1. **학습(Training)**  
    - 입력(X)을 넣고 예측값을 얻어요.  
    - 실제값(y)과 비교해 오차(loss)를 계산해요.  
    - 오차를 줄이도록 모델 내부 숫자(가중치)를 조금씩 바꿔요.
    2. **예측(Inference)**  
    - 배운 모델에 마지막 숫자를 넣으면 다음 숫자를 예측해줘요.

    ##### 5. **실습 팁**
    - **유닛 수**(hidden size)와 **에포크 수**(학습 횟수)를 바꿔보며 결과 변화를 관찰해 보세요.
    - 입력 시퀀스 길이(`timesteps`)를 1보다 크게 늘려보세요.
    - 다른 **최적화 방법**(optimizer)이나 **손실 함수**(loss)를 시도해 보세요.
    '''
    m1_material= r'''
    ##### 📖 학습 목표

    - LSTM 3층 모델로 수열을 예측하는 전체 과정을 이해한다.  
    - `NumPy`, `TensorFlow/Keras`를 활용해 데이터를 준비하고 모델을 구성하는 방법을 익힌다.  
    - Streamlit으로 만든 인터랙티브 앱을 통해 하이퍼파라미터 조절과 코드 실행 과정을 경험한다.  

    ---

    ##### 📘 코드 설명
    ```python
    1) 필요한 도구 불러오기
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    ```
    - numpy: 숫자 계산과 배열을 쉽게 처리하는 도구예요.
    - tensorflow.keras: 딥러닝(신경망)을 만들고 학습할 수 있는 라이브러리예요.
      - Sequential: 층을 순서대로 쌓는 모델
      - LSTM: 수열(시간 순서 있는 데이터)을 잘 처리하는 특별한 층
      - Dense: 마지막에 예측 결과를 출력하는 일반적인 층
    ```python
    2) 입력값 설정
    seq = [2,5,8,11]
    n = 5
    ```
    - `seq`: 우리가 학습시킬 수열입니다. 예: 등차수열 `[2,5,8,11]`
    - `n`: 예측하고 싶은 항 번호입니다. 예: 5번째 항
    ```python
    3) 데이터 전처리
   data = np.array(seq, dtype=float)
    X, y = [], []
    for i in range(len(data) - 1):
        X.append([data[i]])
        y.append(data[i+1])
    X = np.array(X).reshape(-1, 1, 1)
    y = np.array(y) 
    ```
    - `data`: 리스트를 넘파이 배열로 바꿔 계산하기 편하게 만들어요.
    - `X`: 입력값 (현재 숫자), `y`: 정답값 (다음 숫자)
    - `reshape(-1, 1, 1)`: 딥러닝 LSTM이 이해할 수 있는 3차원 구조로 바꿔요
      - 형식: (데이터 개수, 시간 흐름 수, 특징 수) → 여기선 시간도 1, 특징도 1 
    ```python
    4) LSTM 모델 구성(3 Layer)
    model = Sequential([
    LSTM(200, return_sequences=True, input_shape=(1, 1)),
    LSTM(64, return_sequences=False),
    Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    ```
    - 첫 번째 LSTM 층: 200개의 기억 셀을 사용하고 다음 층으로 전체 시퀀스를 전달
    - 두 번째 LSTM 층: 마지막 정보만 전달
    - Dense 층: 하나의 숫자 예측
    - `adam`: 똑똑한 학습 방법 (빠르고 정확함)
    - `mse`: 예측값과 실제값의 차이를 평균제곱으로 계산
    ```python
    5) 모델 학습
    model.fit(X, y, epochs=100, verbose=0)
    ```
    - 모델이 X와 y를 보고 100번 반복 학습합니다.
    ```python
    6) 예측 실행
    last_input = np.array([[[data[-1]]]])
    pred = model.predict(last_input)[0, 0]
    ```
    - `data[-1]`: 학습한 수열의 마지막 숫자 (여기선 11)를 이용해서
    - 다음 항(5번째 항)을 예측합니다.
    - `model.predict(...)`: 예측값을 계산해줍니다.
    - `[0, 0]`: 예측 결과는 배열이라 [0, 0]으로 실제 숫자만 꺼내요.

    ```python
    print()
    print(f"예측된 {n}번째 항: {pred:.2f}")
    ```
    - 예측된 다음 항을 보기 좋게 출력합니다.
    - `:.2f` 는 소수점 둘째 자리까지 깔끔하게 보여주기 위한 형식이에요.
    
    ---

    **💡 실습 팁**

    - 유닛 수와 학습 횟수(에포크)를 바꾸면 모델의 성능이 달라져요.  
    - 수열을 `[1, 2, 4, 7]`처럼 바꿔서 예측이 잘 되는지 시험해보세요.  
    - 예측값과 실제 다음 숫자가 얼마나 가까운지도 비교해 보세요.  
    
    '''

    with st.expander("📚 **딥러닝의 이해**"):
        st.markdown(pre_knowledge)
    with st.expander("📝 **딥러닝 수열 예측 파이썬 코드의 이해**"):
        st.markdown(m1_material)
    st.divider()

    st.markdown("### 🚀 수열 예측 실습")

    # 하이퍼파라미터 입력
    units1 = st.slider("1층 LSTM 유닛 수", 128, 256, 200, 10)
    units2 = st.slider("2층 LSTM 유닛 수", 64, 128, 64, 10)
    epochs = st.slider("학습 에포크 수", 10, 500, 100, 10)

    st.divider()

    # 수열 입력
    seq_input = st.text_input("수열 입력 (콤마로 구분)", "2,5,8,11")
    term_idx = st.number_input("예측할 항 번호", min_value=1, value=len(seq_input.split(",")) + 1)
    
    # "코드 생성하기" 버튼 누를 때마다 즉시 코드 업데이트
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = ""

    if st.button("🔄 코드 생성하기"):
        st.session_state.generated_code = textwrap.dedent(RAW_CODE_TEMPLATE.format(
            seq_input=seq_input,
            term_idx=term_idx,
            units1=units1,
            units2=units2,
            epochs=epochs
        ))
        st.session_state.show_full = True  # 버튼 클릭 시 무조건 코드 보이기 활성화

    # 항상 session_state의 최신 코드를 사용하도록 st_ace 에디터를 호출
    if st.session_state.get("show_full", False):
        st.subheader("📥 실행 코드 (수정 가능)")
        
        # 중요한 부분: st_ace의 key를 유니크하게 유지하여 즉시 업데이트되도록 설정
        user_code = st_ace(
            value=st.session_state.generated_code,
            language="python",
            theme="monokai",
            height=350,
            key=f"ace_lstm_3layer_{st.session_state.generated_code}"
        )

        # 보안 필터링 및 코드 실행
        forbidden_keywords = ["import os", "import shutil", "import subprocess", "open(", "__import__"]
        if st.button("▶️ LSTM 예측 실행하기"):
            if any(keyword in user_code for keyword in forbidden_keywords):
                st.error("⚠️ 위험한 코드는 실행할 수 없습니다.")
            else:
                buf = io.StringIO()
                try:
                    sys.stdout = buf
                    exec(user_code, {})
                except Exception as e:
                    st.error(f"❌ 오류 발생:\n{e}")
                finally:
                    sys.stdout = sys.__stdout__

                result = buf.getvalue().strip()
                if result:
                    st.success(f"🎉 실행 결과:\n```\n{result}\n```")
                else:
                    st.info("ℹ️ 출력 결과가 없습니다.")

if __name__ == "__main__":
    show()