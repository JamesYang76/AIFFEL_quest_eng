#### ■ PART A— 전략과 기획

```
■ 프로젝트 한 줄 정의
AI 예측(Transformer), 기술적 분석(Momentum), 뉴스 감성 분석(NLP)을 결합하여 매매 신호를 생성하고 자동으로 거래를 수행하는 복합 앙상블 트레이딩 시스템

■ ML 도입 판단
이유: 주가는 거시 지표뿐만 아니라 시장의 심리(뉴스)와 현재의 추세(지표)가 복합적으로 작용함 → 단순 모델보다 여러 신호를 통합하는 앙상블 모델이 필요함
접근법:
[✓] Transformer: 시계열 데이터를 통한 14일 후 미래 가격 예측
[✓] Sentiment Analysis: Alpha Vantage API를 활용한 뉴스 데이터 감성 스코어링
[✓] Technical Filter: SMA, MACD, RSI를 활용한 매매 타이밍 정밀 제어
아키타입: [✓] Software 2.0 & Ensemble (복합 모델이 매수/매도 결정을 자동화)
■ 성공 메트릭
비즈니스: 3가지 신호(AI+기술+감성)가 일치하는 'High Confidence' 종목의 수익률 및 손절 라인(-7%) 준수 여부
ML: Transformer 예측 정확도(MAE/Accuracy) 및 뉴스 감성 분석의 상관관계 분석 지표
■ 데이터
소스: FRED(거시 경제), Yahoo Finance(주가), Alpha Vantage(뉴스 감성 데이터), Supabase
규모: 2006년 이후 시계열 데이터 + 실시간 뉴스 피드 데이터
레이블: 14일 후 예상 종가 + 실시간 시장 감성 점수(Average Sentiment Score)
■ 베이스라인: 단일 Transformer 기반 예측 모델 또는 단순 이동평균선 매매
■ 리스크
신호 불일치(Signal Conflict): AI 예측은 상승이나 기술적 지표/뉴스가 하락일 때의 의사결정 복잡성 증가
뉴스 API 의존성: Alpha Vantage와 같은 외부 뉴스 데이터 소스의 지연이나 품질 변화에 따른 감성 점수 왜곡
손절 로직 오작동: 장중 급락 시 API 호출 제한 등으로 인해 -7% 손절 라인 대응이 늦어질 리스크



```
#### 📋 Part B. 인프라와 데이터
```
■ 환경 관리

가상환경: [V] venv [ ] conda [ ] uv
의존성 잠금: [ ] pip-compile [ ] uv lock (현재는 requirements.txt 또는 직접 설치 방식)
핵심 라이브러리: tensorflow, keras, pandas, scikit-learn, yfinance, supabase, requests, schedule
■ 모델 선택

사전 학습 모델: Custom Transformer (Encoder-based Architecture)
선택 이유: 거시 경제 지표와 주가 데이터 사이의 복잡한 비선형 관계 및 장기 의존성(Long-term dependency)을 학습하기 위해 Multi-Head Attention 메커니즘이 가장 유리함
■ GPU / 클라우드

학습 환경: [V] 로컬 GPU (Apple Silicon 가속) [ ] Colab [V] 클라우드 (Supabase DB 연동) [ ] CPU only
예상 학습 시간: 데이터 전처리 포함 약 10~20분 (50 Epochs 기준)
■ 실험 관리

도구: [ ] W&B [ ] MLflow [ ] TensorBoard [V] 수동 기록 (stock_scheduler.log 및 Supabase 테이블)
추적할 메트릭: Train Loss (MSE), MAE, Custom Accuracy (100-MAPE), Rise Probability (%)
■ 데이터 기댓값 테스트

결측치 무결성: 전처리(ffill/bfill) 후 수치형 컬럼에 NaN이나 0 이하의 주가 데이터가 없어야 함
데이터 동기화: FRED 경제 지표와 Yahoo Finance 주가 데이터의 날짜 인덱스가 유실 없이 일치해야 함
예측 신뢰도: 산출된 Accuracy (%) 지표가 논리적 범위(0~100%) 내에 존재해야 함
■ 데이터 버전 관리
현재 레벨: Level 1 (스냅샷 - 매일 최신 데이터를 Supabase에 덮어쓰거나 리샘플링)
목표 레벨: Level 2 (Git LFS 또는 DVC를 도입하여 모델 가중치와 학습 데이터셋 버전 관리)

```

#### 📋 Part C. 테스트와 품질
```
■ 암기 테스트
  overfit_batches=1 테스트를 적용할 수 있는가? [✓] Yes  [ ] No
  No라면 이유: ________________________________________________
  10분 이내로 만들기 위해 조정할 것:
  모델 크기 축소 (Transformer Layer 4개 → 2개), Lookback 기간 축소(90일 → 30일), 배치 수 감소

■ 행동 테스트 (최소 2개 설계)
  불변성 테스트 1: 입력 데이터(주가, 경제 지표)에 미세한 노이즈(Jitter)를 섞어 바꿔도 결과(상승/하락 방향 예측)가 같아야 함
  
  불변성/방향성 테스트 2: 기준금리(FEDFUNDS) 지표를 강제로 대폭 인상된 값으로 입력하면, 빅테크 종목의 미래 예측 가격이 하향 조정되어야 함

■ 계층적 테스트 배치
  PR 단계 (10분 이내): Input/Output Shape 검증, 데이터 전처리 후 결측치(NaN) 및 0 이하 값 검사, 암기 테스트, 린팅(Ruff)
  
  야간/주간 단계: 전체 데이터셋을 활용한 기간별 백테스팅(MAPE 벤치마크), 최근 14일간 실측 데이터와 예측 정확도 추적 검증

■ 트러블슈팅 계획
  Make it Run — 예상되는 구동 오류: 외부 API(FRED, Yahoo Finance, Alpha Vantage)의 호출 제한(Rate Limit) 초과 또는 응답 포맷 변경으로 인한 파싱(Parsing) 오류
  Make it Fast — 예상 병목: [✓] 데이터 로딩  [ ] 모델 연산  [ ] 기타
  Make it Right — 성능이 안 나올 때 첫 번째 시도:
    [ ] 데이터 더 모으기  [ ] 모델 키우기  [✓] 하이퍼파라미터 조정 (Lookback 기간 증감 및 Learning Rate 조정)

■ 평가 전략 (LLM 기반 프로젝트인 경우) *본 프로젝트는 시계열 딥러닝 + Rule 기반
  [✓] 자동 메트릭 (종류: MAPE, MSE, Custom Accuracy(100 - MAPE))
  [ ] LLM-as-a-Judge
  [✓] Human evaluation (빈도: 매일 아침 생성되는 자동 매매 리포트 수동 검토)
  [ ] 골든 세트 (규모: ______건)

■ 모니터링 (배포 후)
  가장 가능성 높은 드리프트 유형: [ ] 입력  [ ] 출력  [✓] 성능
  예상 원인: 시장의 '장세 변화(Regime Change)'. 예컨대 과거의 물가/금리 데이터와 주가 간의 상관관계(커플링 패턴)가 더 이상 현재 시장 기조에서 동일하게 작동하지 않는 디커플링 우려
```

#### 📋 Part D. 배포 연결
```
■ 서빙 방식
  [ ] FastAPI  [ ] Streamlit  [✓] 배치 추론  [ ] 기타: 자체 백그라운드 스케줄러 데몬
■ 실행 설계 (배치/스케줄러 구동)
  동작 방식: 파이썬 `schedule` 스레드를 통해 백그라운드에서 자동 실행
  실행 주기: 매수(한국 기준 00:00 1회 실행) / 매도(미국 장 열릴 시 1분마다 무한 루프)
  입출력: (입력) Supabase 적재 예측값 및 본인 계좌 잔고 → (출력) 한국투자증권(KIS) API 실전 주문 전송
■ 체크포인트에서 모델 로드
  load_from_checkpoint 사용: [ ] Yes  [✓] No 
  (이유: 매매 로직은 모델을 메모리에 직접 올리지 않습니다. 딥러닝 예측(`predict.py`)은 별개의 배치 작업으로 이루어져 DB에 결과만 박아두고, 매매 스케줄러는 가볍게 DB 조회만 해서 빠르게 주문을 쏩니다.)
```



- 이번 프로젝트를 하면서 느낀점, 배운점 \
  기획, 문제점등 모든과정을 전방위로 앓수있어서 좋았다.
- 이번 프로젝트에서 잘 했다고 생각이 드는 점.\
   서비스를 실행할기전에 이런 문제점이 있구나 생각해볼수 있었다.
- 이번 프로젝트에서 느낀 문제점.\
  실제 구현해보기전에는 기획에서 알수 없는 모호함이 있다.
- 다음에는 이렇게 해야겠다 생각한 점.\
  익숙해져야함.
