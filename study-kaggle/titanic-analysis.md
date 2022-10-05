---
description: 레오나르도 디카프리오~
---

# 타이타닉 생존자 분석

먼저 Titanic - Machine Learning from Disaster 를 클릭합니다. 화면에 영어로 된 내용을 읽기 귀찮다면 구글의 번역기능을 활용하세요.

{% embed url="https://www.kaggle.com/competitions/titanic" %}
Titanic
{% endembed %}

머신러닝으로 타이타닉에서 살아남은 승객을 예측하는 모델을 만드는 내용입니다.

타이타닉에 대한 내용은 위키피디아를 통해서 확인바랍니다.

{% embed url="https://ko.wikipedia.org/wiki/RMS_%ED%83%80%EC%9D%B4%ED%83%80%EB%8B%89" %}

<figure><img src="../.gitbook/assets/perso_20221005_003.png" alt=""><figcaption></figcaption></figure>

위와 같이 하위 메뉴로 Overview, Data, Code, Discussion 등의 내용이 존재합니다.

첫번째 챌린지인 타이타닉은 '어떤 종류의 사람들이 생존할 가능성이 더 높았는지'에 대한 답변을 위한 모델을 구축하는 것입니다. 승객 데이터에는 이름, 나이, 성별, 사회적 경제계층 등의 컬럼이 존재합니다.

위의 내용은 Overview 에 나와 있는 내용입니다.

Data라는 하위 메뉴 탭에서 어떠한 데이터가 존재하는 지 알 수 있습니다. 현재는 세개의 데이터가 존재하는데 gender\_submission.csv, train.csv, test.csv 입니다. 오프라인으로 다운로드 받아서 직접 노트북을 운용해도 되지만, 여기서는 커널을 통해 바로 캐글의 개발환경을 사용하도록 하겠습니다.

세번째 Code를 누르면서 커널을 시작할 수 있습니다. 하위메뉴 오른쪽 끝에 'New Notebook'이라는 버튼이 보이면 이를 클릭합니다.

구글 코랩과 같은 화면이 나타납니다.&#x20;

먼저 커널의 제목을 작성합니다. 여기서는 'Titanic prediction - Step by step' 이라고 작성했습니다.

### 분석을 시작합시다

커널 내에 마크다운을 추가하고 아래와 같이 적습니다.

```
# Titanic 

커널을 시작합니다. 아래의 사이트를 참조했습니다.
```

다시 마크다운을 추가하고 아래의 내용을 적습니다

```
## 1 라이브러리 로드
```

아래의 라이브러리 로드 코드는 캐글에서 제공해주는 기본 소스를 그대로 사용해도 무방합니다.

```
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) ...
# You can also write temporary files to /kaggle/temp/, but they won't be saved ...
```

이 부분을 실행하면 결과 메시지가 출력됩니다. 이는 여기 커널의 아래에서 다음과 같은 입력파일을 사용할 수 있다는 얘기입니다.

```
/kaggle/input/titanic/train.csv
/kaggle/input/titanic/test.csv
/kaggle/input/titanic/gender_submission.csv
```

추가적인 라이브러리를 위하여 코드를 추가합니다.

```
# 데이터 시각화 관련
import matplotlib.pyplot as plt
import seaborn as sns
```

마크다운을 추가합니다.

```
## 2 입력데이터 로드 및 확인
```

위에서 나왔던 세가지 입력데이터 중에서 train.csv를 먼저 읽어와 보겠습니다.

```
# 훈련 데이터 로드
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.tail()
```

<figure><img src="../.gitbook/assets/perso_20221005_005.png" alt=""><figcaption><p>train data</p></figcaption></figure>

위와 같이 보면 총 890까지 데이터가 나와 있습니다. 인덱스는 0부터 시작하기 때문에 총 891개의 데이터가 존재하는 것을 알 수 있습니다. 여기에서의 특이점은 Age, Cabin에 NaN이 보이는 것입니다. 값이 없다는 뜻이죠.
