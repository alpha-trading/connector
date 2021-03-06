## 단순이동평균-sma  

단순 이동평균을 구하는 함수입니다.
단순 이동평균은 특정 기간(period) 동안의 가격(price)의 산술평균을 구합니다.  
### 사용 방법
첫 번째 인자에는 단순 이동평균을 구하는데 사용하는 가격을,
두 번째 인자에는 단순 이동평균을 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 5일간 종가의 단순 이동평균을 구하고자 하는 경우
`sma(close, 5)` 또는 `단순이동평균(종가, 5)`와 같이 작성하면 됩니다.  
### 계산 방법
5일간 종가의 단순 이동평균은 다음과 같이 구합니다.
(당일 종가 + 1일 전 종가 + 2일 전 종가 + 3일 전 종가 + 4일 전 종가) / 5  
```python
sma(price, period)
단순이동평균(가격데이터, 기간)
```
### 변수
`price:  단순 이동평균을 구하는데 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  단순 이동평균을 구하는데 사용하는 기간`  

## 지수이동평균-ema  

지수 이동평균을 구하는 함수입니다.
지수 이동평균은 가격(price)의 최근 데이터에 가중치를 두어 평균을 구합니다.  
<사용 법>
첫 번째 인자에는 지수 이동평균을 구하는데 사용하는 가격을,
두 번째 인자에는 지수 이동평균을 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 5일간 종가의 지수 이동평균을 구하고자 하는 경우
`ema(close, 5)` 또는 `지수이동평균(종가, 5)`와 같이 작성하면 됩니다.  
### 계산 방법
5일간 종가의 지수 이동평균은 다음과 같이 구합니다.
당일 지수 이동평균 = 당일 종가 * 평활 계수 + 전일 지수 이동평균 * (1 - 평활 계수)
평활 계수 = 2 / (5 + 1)  
```python
ema(price, period)
지수이동평균(가격데이터, 기간)
```
### 변수
`price:  지수 이동평균을 구하는데 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  지수 이동평균을 구하는데 사용하는 기간`  

## 가중이동평균-wma  

가중 이동평균을 구하는 함수입니다.
가중 이동평균은 평균을 구하는 데 있어서 주어지는 가중값을 반영시킵니다.  
### 사용 방법
첫 번째 인자에는 가중 이동평균을 구하는데 사용하는 가격을,
두 번째 인자에는 가중 이동평균을 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 5일간 종가의 가중 이동평균을 구하고자 하는 경우
`wma(close, 5)` 또는 `가중이동평균(종가, 5)`와 같이 작성하면 됩니다.  
### 계산 방법
5일간 종가의 가중 이동평균은 다음과 같이 구합니다.
(당일 종가 * 5 + 1일 전 종가 * 4 + 2일 전 종가 * 3 + 3일 전 종가 * 2 + 4일 전 종가 * 1) / (5 + 4 + 3 + 2 + 1)  
```python
wma(price, period)
가중이동평균(가격데이터, 기간)
```
### 변수
`price:  가중 이동평균을 구하는데 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  가중 이동평균을 구하는데 사용하는 기간`  

## 볼린저밴드상한선-bollinger_upper  

볼린저 밴드 상한선(상향 밴드)을 구하는 함수입니다.
볼린저 밴드의 상한선은 표준 편차에 의해 산출된 이동평균 값이며,
주가나 지수의 움직임이 큰 시기에는 밴드의 폭이 넓어지고 움직임이 작은 시기에는 밴드의 폭이 좁아지는 특성을 가지고 있습니다.
즉, 가격 움직임의 크기에 따라 밴드의 넓이가 결정되는 것입니다.  
### 사용 방법
첫 번째 인자에는 상향 밴드를 구하는데 사용하는 가격을,
두 번째 인자에는 상향 밴드를 구하는데 사용하는 기간을,
세 번째 인자에는 중간 밴드를 구하는데 사용하는 이동평균 종류를,
네 번째 인자에는 상향 밴드를 구할 때 사용하는 표준편차 승수를 적으면 됩니다.
예를 들어, 20일간 종가의 단순 이동평균으로 중간 밴드를 구하고 상향 밴드는 중간 밴드에 20일간 종가의 표준편차에 2배를 더한 값을 사용하고자 할 경우
`bollinger_upper(close, 20, sma, 2)` 또는 `볼린저밴드상한선(종가, 20, 단순이동평균, 2)`와 같이 작성하면 됩니다.  
```python
bollinger_upper(price, period, moving_average, multiplier)
볼린저밴드상한선(가격데이터, 기간, 이동평균종류, 승수값)
```
### 변수
`price:  상향 밴드를 구하는데 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  상향 밴드를 구하는데 사용하는 기간`  
`moving_average:  중간 밴드를 구할 때 사용하는 이동평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  
`multiplier:  상향 밴드를 구할 때 사용하는 표준편차 승수`  

## 볼린저밴드하한선-bollinger_lower  

볼린저 밴드 하한선(하향 밴드)을 구하는 함수입니다.
볼린저 밴드의 하한선은 표준 편차에 의해 산출된 이동평균 값이며,
주가나 지수의 움직임이 큰 시기에는 밴드의 폭이 넓어지고 움직임이 작은 시기에는 밴드의 폭이 좁아지는 특성을 가지고 있습니다.
즉, 가격 움직임의 크기에 따라 밴드의 넓이가 결정되는 것입니다.  
### 사용 방법
첫 번째 인자에는 하향 밴드를 구하는데 사용하는 가격을,
두 번째 인자에는 하향 밴드를 구하는데 사용하는 기간을,
세 번째 인자에는 중간 밴드를 구하는데 사용하는 이동평균 종류를,
네 번째 인자에는 하향 밴드를 구할 때 사용하는 표준편차 승수를 적으면 됩니다.
예를 들어, 20일간 종가의 단순 이동평균으로 중간 밴드를 구하고 하향 밴드는 중간 밴드에 20일간 종가의 표준편차에 2배를 뺀 값을 사용하고자 할 경우
`bollinger_lower(close, 20, sma, 2)` 또는 `볼린저밴드하한선(종가, 20, 단순이동평균, 2)`와 같이 작성하면 됩니다.  
```python
bollinger_lower(price, period, moving_average, multiplier)
볼린저밴드하한선(가격데이터, 기간, 이동평균종류, 승수값)
```
### 변수
`price:  하향 밴드를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  하향 밴드를 구하는데 사용하는 기간`  
`moving_average:  중간 밴드를 구할 때 사용하는 이동평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  
`multiplier:  하향 밴드를 구할 때 사용하는 표준편차 승수`  

## 엔벨로프상한선-envelope_upper  

엔벨로프 상한선을 구하는 함수입니다.
엔벨로프 상한선은 주가의 이동평균 선에서 일정 비율 만큼 더한 선입니다.  
### 사용 방법
첫 번째 인자에는 엔벨로프 상한선을 구하는데 사용하는 가격을,
두 번째 인자에는 엔벨로프 상한선을 구하는데 사용하는 기간을,
세 번째 인자에는 엔벨로프 상한선을 구하는데 사용하는 이동평균 종류를,
네 번째 인자에는 엔벨로프 상한선을 구할 때 사용하는 비율을 적으면 됩니다.
예를 들어, 20일간 종가의 단순 이동평균으로 기준선을 구하고 엔벨로프 상한선은 기준선에서 8% 위의 선으로 하고자 하는 경우
`envelope_upper(close, 20, sma, 0.08)` 또는 엔벨로프상한선(종가, 20, 단순이동평균, 0.08)`과 같이 작성하면 됩니다.  
```python
envelope_upper(price, period, moving_average, ratio)
엔벨로프상한선(가격데이터, 기간, 이동평균종류, 비율)
```
### 변수
`price:  엔벨로프 상한선을 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  엔벨로프 상한선을 구하는데 사용하는 기간`  
`moving_average:  엔벨로프 상한선을 구하는데 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  
`ratio:  엔벨로프 상한선을 구하는데 사용하는 비율`  

## 엔벨로프하한선-envelope_lower  

엔벨로프 하한선을 구하는 함수입니다.
엔벨로프 하한선은 주가의 이동평균 선에서 일정 비율 만큼 뺀 선입니다.  
### 사용 방법
첫 번째 인자에는 엔벨로프 하한선을 구하는데 사용하는 가격을,
두 번째 인자에는 엔벨로프 하한선을 구하는데 사용하는 기간을,
세 번째 인자에는 엔벨로프 하한선을 구하는데 사용하는 이동평균 종류를,
네 번째 인자에는 엔벨로프 하한선을 구할 때 사용하는 비율을 적으면 됩니다.
예를 들어, 20일간 종가의 단순 이동평균으로 기준선을 구하고 엔벨로프 하한선은 기준선에서 8% 아래의 선으로 하고자 하는 경우
`envelope_lower(close, 20, sma, 0.08)` 또는 `엔벨로프하한선(종가, 20, 단순이동평균, 0.08)`과 같이 작성하면 됩니다.  
```python
envelope_lower(price, period, moving_average, ratio)
엔벨로프하한선(가격데이터, 기간, 이동평균종류, 비율)
```
### 변수
`price:  엔벨로프 하한선을 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  엔벨로프 하한선을 구하는 기간`  
`moving_average:  엔벨로프 하한선을 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  
`ratio:  엔벨로프 상한선을 구하는데 사용하는 비율`  

## 가격채널상한선-price_channel_upper  

가격 채널 상한선을 구하는 함수입니다.
가격 채널 상한선은 일정 기간 내의 최고가를 이은 선입니다.  
### 사용 방법
첫 번째 인자에는 고가를,
두 번째 인자에는 가격 채널 상한선을 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 20일간 채널 지표 상한선을 구하고자 하는 경우
`price_channel_upper(high, 20)` 또는 `가격채널상한선(고가, 20)`과 같이 작성하면 됩니다.  
```python
price_channel_upper(price_high, period)
가격채널상한선(고가, 기간)
```
### 변수
`price_high:  고가`  
`period:  가격 채널 상한선을 구할 때 사용하는 기간`  

## 가격채널하한선-price_channel_lower  

가격 채널 하한선을 구하는 함수입니다.
가격 채널 하한선은 일정 기간 내의 최저가를 이은 선입니다.  
### 사용 방법
첫 번째 인자에는 저가를,
두 번째 인자에는 가격 채널 하한선을 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 20일간 채널 지표 하한선을 구하고자 하는 경우
`price_channel_lower(low, 20)` 또는 `가격채널하한선(저가, 20)`과 같이 작성하면 됩니다.  
```python
price_channel_lower(price_low, period)
가격채널하한선(저가, 기간)
```
### 변수
`price_low:  저가`  
`period:  가격 채널 상한선을 구하는 기간`  

## 매수방향지표-pdi  

매수방향지표(PDI)를 구하는 함수입니다.
매수방향지표(PDI)는 실질적으로 상승하는 폭의 비율을 나타냅니다.
매수방향지표(PDI)는 0에서 1사이의 값으로 표현됩니다.  
### 사용 방법
첫 번째 인자에는 고가를,
두 번째 인자에는 저가를,
세 번째 인자에는 종가를,
네 번째 인자에는 매수방향지표(PDI)를 구하는데 사용하는 기간을,
다섯 번째 인자에는 매수방향지표(PDI)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 지수 이동 평균을 이용한 14일간 매수방향지표(PDI)를 구하고자 하는 경우
`pdi(high, low, close, 14, ema)` 또는 `매수방향지표(고가, 저가, 종가, 14, 지수이동평균)`과 같이 작성하면 됩니다.  
```python
pdi(price_high, price_low, price_close, period, moving_average)
매수방향지표(고가, 저가, 종가, 기간, 이동평균종류)
```
### 변수
`price_high:  고가`  
`price_low:  저가`  
`price_close:  종가`  
`period:  매수방향지표(PDI)를 구하는데 사용하는 기간`  
`moving_average:  매수방향지표(PDI)를 구하는데 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 매도방향지표-mdi  

매도방향지표(MDI)를 구하는 함수입니다.
매도방향지표(MDI)는 실질적으로 하락하는 폭의 비율을 나타냅니다.
매도방향지표(MDI)는 0에서 1사이의 값으로 표현됩니다.  
### 사용 방법
첫 번째 인자에는 고가를,
두 번째 인자에는 저가를,
세 번째 인자에는 종가를,
네 번째 인자에는 매도방향지표(MDI)를 구하는데 사용하는 기간을,
다섯 번째 인자에는 매도방향지표(MDI)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 지수 이동 평균을 이용한 14일간 매도방향지표(MDI)를 구하고자 하는 경우
`mdi(high, low, close, 14, ema)` 또는 `매도방향지표(고가, 저가, 종가, 14, 지수이동평균)`과 같이 작성하면 됩니다.  
```python
mdi(price_high, price_low, price_close, period, moving_average)
매도방향지표(고가, 저가, 종가, 기간, 이동평균종류)
```
### 변수
`price_high:  고가`  
`price_low:  저가`  
`price_close:  종가`  
`period:  매도방향지표(MDI)를 구하는데 사용하는 기간`  
`moving_average:  매도방향지표(MDI)를 구하는데 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 평균방향이동지표-adx  

평균방향이동지표(ADX)를 구하는 함수입니다.
평균방향이동지표(ADX)는 추세의 강도를 의미합니다.
평균방향이동지표(ADX)는 0에서 1사이의 값으로 표현됩니다.  
### 사용 방법
첫 번째 인자에는 고가를,
두 번째 인자에는 저가를,
세 번째 인자에는 종가를,
네 번째 인자에는 ADX를 구하는데 사용하는 기간을,
다섯 번째 인자에는 ADX를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 지수 이동 평균을 이용한 14일간 ADX를 구하고자 하는 경우
`adx(high, low, close, 14, ema)` 또는 `평균방향이동지표(고가, 저가, 종가, 14, 지수이동평균)`과 같이 작성하면 됩니다.  
```python
adx(price_high, price_low, price_close, period, moving_average)
평균방향이동지표(고가, 저가, 종가, 기간, 이동평균종류)
```
### 변수
`price_high:  고가`  
`price_low:  저가`  
`price_close:  종가`  
`period:  ADX를 구하는데 사용하는 기간`  
`moving_average:  ADX를 구하는데 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 이동평균수렴확산지수-macd  

이동평균수렴확산지수(MACD)를 구하는 함수입니다.
이동평균수렴확산지수(MACD)는 단기 이동평균 값과 장기 이동평균 값의 차이를 이용한 지표입니다.  
### 사용 방법
첫 번째 인자에는 이동평균수렴확산지수(MACD)를 구하는데 사용하는 가격을,
두 번째 인자에는 단기 이동 평균을 구하는데 사용하는 기간을,
세 번째 인자에는 장기 이동 평균을 구하는데 사용하는 기간을,
네 번째 인자에는 이동평균수렴확산지수(MACD)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 12일간 종가의 단순 이동 평균과 26일간 종가의 단순 이동 평균을 이용하여 이동평균수렴확산지수(MACD)를 구하고자 하는 경우에는
`macd(close, 12, 26, sma)` 또는 `이동평균수렴확산지수(종가, 12, 26, 단순이동평균)`과 같이 작성하면 됩니다.  
```python
macd(price, short_period, long_period, moving_average)
이동평균수렴확산지수(가격데이터, 단기이동평균기간, 장기이동평균기간, 이동평균종류)
```
### 변수
`price:  이동평균수렴확산지수(MACD)를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`short_period:  단기 이동 평균을 구하는데 사용하는 기간`  
`long_period:  장기 이동 평균을 구하는데 사용하는 기간`  
`moving_average:  이동평균수렴확산지수(MACD)를 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 이동평균수렴확산시그널-macd_signal  

이동평균수렴확산시그널(MACD_signal)을 구하는 함수입니다.
이동평균수렴확산시그널(MACD_signal)은 이동평균수렴확산지수(MACD)의 일정 기간 동안의 평균입니다.  
### 사용 방법
첫 번째 인자에는 이동평균수렴확산시그널(MACD_signal)을 구하는데 사용하는 가격을,
두 번째 인자에는 단기 이동 평균을 구하는데 사용하는 기간을,
세 번째 인자에는 장기 이동 평균을 구하는데 사용하는 기간을,
네 번째 인자에는 시그널 기간을,
다섯 번째 인자에는 이동평균수렴확산시그널(MACD_signal)을 구하는데 이용할 이동 평균 종류를 적으면 됩니다.
예를 들어, 12일간 종가의 단순 이동 평균, 26일간 종가의 단순 이동 평균, 9일의 signal 기간을 이용하여
이동평균수렴확산시그널(MACD_signal)을 구하고자 하는 경우에는
`macd_signal(close, 12, 26, 9, sma)` 또는 `이동평균수렴확산시그널(종가, 12, 26, 9, 단순이동평균)`과 같이 작성하면 됩니다.  
```python
macd_signal(price, short_period, long_period, signal_period, moving_average)
이동평균수렴확산시그널(기간, 단기이동평균기간, 장기이동평균기간, 시그널기간, 이동평균종류)
```
### 변수
`price:  이동평균수렴확산시그널(MACD_signal)를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`short_period:  단기 이동 평균을 구하는데 사용하는 기간`  
`long_period:  장기 이동 평균을 구하는데 사용하는 기간`  
`signal_period:  이동평균수렴확산시그널(MACD_signal)를 구할 때 사용하는 시그널 기간`  
`moving_average:  이동평균수렴확산시그널(MACD_signal)를 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 이동평균수렴확산오실레이터-macd_oscillator  

이동평균수렴확산오실레이터(MACD_oscillator)를 구하는 함수입니다.
이동평균수렴확산오실레이터(MACD_oscillator)는 MACD와 Signal의 차를 통해 계산됩니다.  
### 사용 방법
첫 번째 인자에는 이동평균수렴확산오실레이터(MACD_oscillator)를 구하는데 사용하는 가격을,
두 번째 인자에는 단기 이동 평균을 구하는데 사용하는 기간을,
세 번째 인자에는 장기 이동 평균을 구하는데 사용하는 기간을,
네 번째 인자에는 시그널 기간을,
다섯 번째 인자에는 이동평균수렴확산오실레이터(MACD_oscillator)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 12일간 종가의 단순 이동 평균, 26일간 종가의 단순 이동 평균, 9일의 signal 기간을 이용하여
이동평균수렴확산오실레이터(MACD_oscillator)를 구하고자 하는 경우에는
`macd_oscillator(close, 12, 26, 9, sma)` 또는 `이동평균수렴확산오실레이터(종가, 12, 26, 9, 단순이동평균)`과 같이 작성하면 됩니다.  
```python
macd_oscillator(price, short_period, long_period, signal_period, moving_average)
이동평균수렴확산오실레이터(가격데이터, 단기이동평균기간, 장기이동평균기간, 시그널기간, 이동평균종류)
```
### 변수
`price:  이동평균수렴확산오실레이터(MACD_oscillator)를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`short_period:  단기 이동 평균을 구하는데 사용하는 기간`  
`long_period:  장기 이동 평균을 구하는데 사용하는 기간`  
`signal_period:  이동평균수렴확산오실레이터(MACD_oscillator)를 구할 때 사용하는 시그널 기간`  
`moving_average:  이동평균수렴확산오실레이터(MACD_oscillator)를 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 거래량비율-volume_ratio  

거래량비율(Volume Ratio)을 구하는 함수입니다.
거래량비율(Volume Ratio)은 일정 기간 동안의 상승일의 거래량과 하락일의 거래량을 비교합니다.
거래량비율(Volume Ratio)은 0에서 1사이의 값으로 표현됩니다.  
### 사용 방법
첫 번째 인자에는 거래량비율(Volume Ratio)을 구하는데 사용하는 가격을,
두 번째 인자에는 거래량을,
세 번째 인자에는 거래량비율(Volume Ratio)을 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 20일간의 종가를 이용한 거래량비율(Volume Ratio)을 구하고자 하는 경우에는
`volume_ratio(close, volume, 20)` 또는 `거래량비율(종가, 거래량, 20)`과 같이 작성하면 됩니다.  
```python
volume_ratio(price, volume, period)
거래량비율(가격데이터, 거래량, 기간)
```
### 변수
`price:  거래량비율(Volume Ratio)을 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`volume:  거래량`  
`period:  거래량비율(Volume Ratio)을 구하는데 사용하는 기간`  

## 투자심리도-psychological_line  

투자심리도(Psychological Line)를 구하는 함수입니다.
투자심리도(Psychological Line)를 이용하면 과열 및 침체도를 파악할 수 있습니다.
투자심리도(Psychological Line)는 0에서 1사이의 값으로 표현됩니다.  
### 사용 방법
첫 번째 인자에는 투자심리도(Psychological Line)를 구하는데 사용하는 가격을,
두 번째 인자에는 투자심리도(Psychological Line)를 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 10일간의 종가를 이용한 투자심리도(Psychological Line)를 구하고자 하는 경우에는
`psychological_line(close, 10)` 또는 `투자심리도(종가, 10)`과 같이 작성하면 됩니다.  
### 계산 방법
10일간의 종가를 이용한 투자심리도(Psychological Line)는 다음과 같이 구합니다.
(10일간 전일 종가 대비 상승 일수) / 10  
```python
psychological_line(price, period)
투자심리도(가격데이터, 기간)
```
### 변수
`price:  투자심리도(Psychological Line)를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  투자심리도(Psychological Line)를 구하는데 사용하는 기간`  

## 신심리도-new_psychological_line  

신심리도(Psychological Line)를 구하는 함수입니다.
신심리도(New Psychological Line)는 주가 등락 폭을 반영하지 못하는 투자심리도(Psychological Line)의 단점을 개선하였습니다.  
### 사용 방법
첫 번째 인자에는 신심리도(New Psychological Line)를 구하는데 사용하는 가격을,
두 번째 인자에는 신심리도(New Psychological Line)를 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 10일간의 종가를 이용한 신심리도(New Psychological Line)를 구하고자 하는 경우에는
`new_psychological_line(close, 10)` 또는 `신심리도(종가, 10)`과 같이 작성하면 됩니다.  
```python
new_psychological_line(price, period)
신심리도(가격데이터, 기간)
```
### 변수
`price:  신심리도(New Psychological Line)를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  신심리도(New Psychological Line)를 구하는데 사용하는 기간`  

## 이격도-disparity  

이격도(Disparity)를 구하는 함수입니다.
이격도(Disparity)는 주가가 이동 평균과 어느 정도 떨어져 있는가 나타냅니다.  
### 사용 방법
첫 번째 인자에는 이격도(Disparity)를 구하는데 사용하는 가격을,
두 번째 인자에는 이격도(Disparity)를 구하는데 사용하는 기간을,
세 번째 인자에는 이격도(Disparity)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 종가와 5일간의 단순 이동평균을 이용한 이격도(Disparity)를 구하고자 하는 경우에는
`disparity(close, 5, sma)` 또는 `이격도(종가, 5, 단순이동평균)`과 같이 작성하면 됩니다.  
```python
disparity(price, period, moving_average)
이격도(가격데이터, 기간, 이동평균종류)
```
### 변수
`price:  이격도(Disparity)를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  이격도(Disparity)를 구하는데 사용하는 기간`  
`moving_average:  이격도(Disparity)를 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 종가위치비율-ibs  

종가위치비율(IBS)을 구하는 함수입니다.
종가위치비율(IBS)은 종가가 당일 변동폭에서 어떠한 지점에 위치해있는지를 나타냅니다.  
### 사용 방법
첫 번째 인자에는 고가를,
두 번째 인자에는 저가,
세 번째 인자에는 종가를 적으면 됩니다.
종가위치비율(IBS)을 구하고자 하는 경우에는
`ibs(high, low, close)` 또는 `종가위치비율(고가, 저가, 종가)`와 같이 작성하면 됩니다.  
### 계산 방법
(종가 - 저가) / (고가 - 저가)  
```python
ibs(price_high, price_low, price_close)
종가위치비율(고가, 저가, 종가)
```
### 변수
`price_high:  고가`  
`price_low:  저가`  
`price_close:  종가`  

## 윗꼬리비율-upper_tail_ratio  

윗꼬리비율(Upper Tail Ratio)을 구하는 함수입니다.  
### 사용 방법
첫 번째 인자에는 시가를,
두 번째 인자에는 고가를,
세 번째 인자에는 종가를 적으면 됩니다.
윗꼬리비율은 `upper_tail_ratio(open, high, close)` 또는 `윗꼬리비율(시가, 고가, 종가)`과 같이 작성하면 됩니다.  
```python
upper_tail_ratio(price_open, price_high, price_close)
윗꼬리비율(시가, 고가, 종가)
```
### 변수
`price_open:  시가`  
`price_high:  고가`  
`price_close:  종가`  

## 아랫꼬리비율-lower_tail_ratio  

아랫꼬리비율(Lower Tail Ratio)을 구하는 함수입니다.  
### 사용 방법
첫 번째 인자에는 시가를,
두 번째 인자에는 저가를,
세 번째 인자에는 종가를 적으면 됩니다.
아랫꼬리비율을 구하고자 하는 경우에는
`lower_tail_ratio(open, low, close)` 또는 `아랫꼬리비율(시가, 저가, 종가)`과 같이 작성하면 됩니다.  
```python
lower_tail_ratio(price_open, price_low, price_close)
아랫꼬리비율(시가, 저가, 종가)
```
### 변수
`price_open:  시가`  
`price_low:  저가`  
`price_close:  종가`  

## 에이비율-a_ratio  

A Ratio를 구하는 함수입니다.
A Ratio는 주가 변동을 이용하여 강, 약 에너지를 파악합니다.  
### 사용 방법
첫 번째 인자에는 시가를,
두 번째 인자에는 고가를,
세 번째 인자에는 저가를,
네 번째 인자에는 A Ratio를 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 26일간의 A Ratio를 구하고자 하는 경우에는
`a_ratio(open, high, low, 26)` 또는 `에이비율(시가, 고자, 저가, 26)`과 같이 작성하면 됩니다.  
```python
a_ratio(price_open, price_high, price_low, period)
에이비율(시가, 고가, 저가, 기간)
```
### 변수
`price_open:  시가`  
`price_high:  고가`  
`price_low:  저가`  
`period:  A Ratio를 구하는데 사용하는 기간`  

## 비비율-b_ratio  

B Ratio를 구하는 함수입니다.
B Ratio는 주가 변동을 이용하여 강, 약 에너지를 파악합니다.  
### 사용 방법
첫 번째 인자에는 고가를,
두 번째 인자에는 저가를,
세 번째 인자에는 종가를,
네 번째 인자에는 B Ratio를 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 26일간의 B Ratio를 구하고자 하는 경우에는
`b_ratio(high, low, close, 26)` 또는 `비비율(고가, 저가, 종가, 26)`과 같이 작성하면 됩니다.  
```python
b_ratio(price_high, price_low, price_close, period)
비비율(고가, 저가, 종가, 기간)
```
### 변수
`price_high:  고가`  
`price_low:  저가`  
`price_close:  종가`  
`period:  B Ratio를 구하는데 사용하는 기간`  

## 질량지수-mass_index  

질량지수(Mass Index)를 구하는 함수입니다.
질량지수(Mass Index)는 고가와 저가 사이의 변동폭을 측정하여 단기적인 추세의 전환점을 찾아냅니다.  
### 사용 방법
첫 번째 인자에는 고가를,
두 번째 인자에는 저가를,
세 번째 인자에는 질량지수(Mass Index)를 구하는데 사용하는 기간을,
네 번째 인자에는 질량지수(Mass Index)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 단순 이동평균을 이용한 25일간 질량지수(Mass Index)를 구하고자 하는 경우에는
`mass_index(high, low, 25, sma)` 또는 `질량지수(고가, 저가, 25, 단순이동평균)`과 같이 작성하면 됩니다.  
```python
mass_index(price_high, price_low, period, moving_average)
질량지수(고가, 저가, 기간, 이동평균종류)
```
### 변수
`price_high:  고가`  
`price_low:  저가`  
`period:  질량지수(Mass Index)를 구하는데 사용하는 기간`  
`moving_average:  질량지수(Mass Index)를 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 이동평균오실레이터-mao  

이동평균오실레이터(MAO)를 구하는 함수입니다.
이동평균오실레이터(MAO)는 단기 이동 평균 값과 장기 이동 평균 값의 차를 통해 계산되며 주가의 추세를 판단할 수 있습니다.  
### 사용 방법
첫 번째 인자에는 이동평균오실레이터(MAO)를 구하는데 사용하는 가격을,
두 번째 인자에는 단기 이동 평균을 구하는데 사용하는 기간을,
세 번째 인자에는 장기 이동 평균을 구하는데 사용하는 기간을,
네 번째 인자에는 이동평균오실레이터(MAO)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 5일간 종가의 단순 이동 평균과 10일간 종가의 단순 이동 평균을 사용하여 이동평균오실레이터(MAO)를 구하고자 하는 경우에는
`mao(close, 5, 10, sma)` 또는 `이동평균오실레이터(종가, 5, 10, 단순이동평균)`과 같이 작성하면 됩니다.  
```python
mao(price, short_period, long_period, moving_average)
이동평균오실레이터(기간, 단기이동평균기간, 장기이동평균기간, 이동평균종류)
```
### 변수
`price:  이동평균오실레이터(MAO)를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`short_period:  단기 이동 평균을 구하는데 사용하는 기간`  
`long_period:  장기 이동 평균을 구하는데 사용하는 기간`  
`moving_average:  이동평균오실레이터(MAO)를 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 소나-sonar  

소나(Sonar)를 구하는 함수입니다.
소나(Sonar)는 주가의 추세 전환 시점을 파악하기 위한 지표입니다.  
### 사용 방법
첫 번째 인자에는 소나(Sonar)를 구하는데 사용하는 가격을,
두 번째 인자에는 이동 평균을 구하는데 사용하는 기간을,
세 번째 인자에는 소나(Sonar)를 구하는데 사용하는 과거 이동 평균 값의 기간을,
네 번째 인자에는 소나(Sonar)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 20일간 종가의 지수 이동 평균과 9일전 지수 이동 평균을 이용하여 소나(Sonar)를 구하고자 하는 경우에는
`sonar(close, 20, 9, ema)` 또는 `소나(종가, 20, 9, 지수이동평균)`과 같이 작성하면 됩니다.  
```python
sonar(price, period, sonar_period, moving_average)
소나(가격데이터, 기간, 소나기간, 이동평균종류)
```
### 변수
`price:  소나(Sonar)를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  이동 평균을 구하는데 사용하는 기간`  
`sonar_period:  사용하고자 하는 과거 이동 평균 값의 기간`  
`moving_average:  소나(Sonar)를 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 소나시그널-sonar_signal  

소나시그널(Sonar Signal)을 구하는 함수입니다.
소나시그널(Sonar Signal)은 주가의 추세 전환 시점을 파악하기 위한 지표입니다.  
### 사용 방법
첫 번째 인자에는 소나(Sonar)를 구하는데 사용하는 가격을,
두 번째 인자에는 이동 평균을 구하는데 사용하는 기간을,
세 번째 인자에는 소나(Sonar)를 구하는데 사용하는 과거 이동 평균 값의 기간을,
네 번째 인자에는 소나시그널(Sonar Signal)을 구하는데 사용하는 소나(Sonar)의 이동 평균 기간을,
네 번째 인자에는 소나(Sonar)를 구하는데 이용할 이동 평균 종류를 적으면 됩니다.
예를 들어, 20일간 종가의 지수 이동 평균과 9일전 지수 이동 평균을 이용하여 시그널 기간이 9인 소나시그널(Sonar Signal)를 구하고자 하는 경우에는
`sonar_signal(close, 20, 9, 9, ema)` 또는 `소나시그널(종가, 20, 9, 9, 지수이동평균)`과 같이 작성하면 됩니다.  
```python
sonar_signal(price, period, sonar_period, signal_period, moving_average)
소나시그널(가격데이터, 기간, 소나기간, 시그널기간, 이동평균종류)
```
### 변수
`price:  소나(Sonar)를 구할 때 사용하는 가격 ex) 시가, 고가, 저가, 종가`  
`period:  이동 평균을 구하는데 사용하는 기간`  
`sonar_period:  사용하고자 하는 과거 이동 평균 값의 기간`  
`signal_period:  소나시그널(Sonar Signal)을 구하는데 사용하는 소나(Sonar)의 이동 평균 기간`  
`moving_average:  소나(Sonar)를 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 자금흐름지수-mfi  

자금흐름지수(MFI)를 구하는 함수입니다.
자금흐름지수(MFI)는 주식시장으로 자금이 유입되거나 유출되는 양을 측정합니다.  
### 사용 방법
첫 번째 인자에는 고가를,
두 번째 인자에는 저가를,
세 번째 인자에는 종가를,
네 번째 인자에는 거래량을,
네 번째 인자에는 자금흐름지수(MFI)를 구하는데 사용하는 기간을 적으면 됩니다.
예를 들어, 14일간 자금흐름지수(MFI)를 구하고자 하는 경우에는
`mfi(high, low, close, volume, 14)` 또는 `자금흐름지수(고가, 저가, 종가, 거래량, 14)`과 같이 작성하면 됩니다.  
```python
mfi(price_high, price_low, price_close, vol, period)
자금흐름지수(고가, 저가, 종가, 거래량, 기간)
```
### 변수
`price_high:  고가`  
`price_low:  저가`  
`price_close:  종가`  
`vol:  거래량`  
`period:  자금흐름지수(MFI)를 구하는데 사용하는 기간`  

## 트라이엄프엑스-trix  

트라이엄프엑스(TRIX)를 구하는 함수입니다.
트라이엄프엑스(TRIX)는 이동 평균을 세 번 구한 후 이 값의 전일 대비 상승비율을 계산합니다.  
### 사용 방법
첫 번째 인자에는 트라이엄프엑스(TRIX)를 구하는데 사용하는 가격을,
두 번째 인자에는 트라이엄프엑스(TRIX)를 구하는데 사용하는 기간을,
세 번째 인자에는 트라이엄프엑스(TRIX)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 10일간 종가의 지수 이동 평균으로 트라이엄프엑스(TRIX)를 구하고자 하는 경우에는
`trix(close, 10, ema)` 또는 `소나시그널(종가, 10, 지수이동평균)`과 같이 작성하면 됩니다.  
```python
trix(price, period, moving_average)
트라이엄프엑스(가격데이터, 기간, 이동평균종류)
```
### 변수
`price:  트라이엄프엑스(TRIX)를 구하는데 사용하는 가격`  
`period:  트라이엄프엑스(TRIX)를 구하는데 사용하는 기간`  
`moving_average:  트라이엄프엑스(TRIX)를 구할 때 사용하는 이동 평균 종류 ex) 단순 이동평균, 지수 이동평균, 가중 이동평균`  

## 트라이엄프엑스시그널-trix_signal  

트라이엄프엑스시그널(TRIX Signal)을 구하는 함수입니다.
트라이엄프엑스시그널(TRIX Signal)은 트라이엄프엑스(TRIX)의 이동 평균 값입니다.  
### 사용 방법
첫 번째 인자에는 트라이엄프엑스(TRIX)를 구하는데 사용하는 가격을,
두 번째 인자에는 트라이엄프엑스(TRIX)를 구하는데 사용하는 기간을,
세 번째 인자에는 트라이엄프엑스시그널(TRIX Signal)을 구하는데 사용하는 시그널 기간을,
네 번째 인자에는 트라이엄프엑스(TRIX)를 구하는데 사용하는 이동 평균 종류를 적으면 됩니다.
예를 들어, 10일간 종가의 지수 이동 평균으로 트라이엄프엑스(TRIX)를 구하고 9일간 트라이엄프엑스(TRIX)의 지수 이동 평균을 구하고자 하는 경우에는
`trix_signal(close, 10, 9, ema)` 또는 `소나시그널(종가, 10, 9, 지수이동평균)`과 같이 작성하면 됩니다.  
```python
trix_signal(price, period, signal_period, moving_average)
트라이엄프엑스시그널(가격데이터, 기간, 시그널기간, 이동평균종류)
```
### 변수
`price:  트라이엄프엑스(TRIX)를 구하는데 사용하는 가격`  
`period:  트라이엄프엑스(TRIX)를 구하는데 사용하는 기간`  
`signal_period:  트라이엄프엑스시그널(TRIX Signal)을 구하는데 사용하는 시그널 기간`  
`moving_average:  트라이엄프엑스(TRIX)를 구하는데 사용하는 이동 평균 종류`  
