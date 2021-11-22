import numpy as np

# 전처리 함수 : 전체 데이터 프레임에서 가장 마지막 행을 기준으로 시퀀스 데이터 구성
def data2seq(df, length = 4) :

    # 결과 저장을 위한 배열 생성
    res = np.zeros((1, length, 2))

    # 스케일링을 위한 값 설정
    v_min = 343.6
    v_max = 360.0
    c_min = 200.0
    c_max = 1068.0

    # 결과 저장을 위한 배열 생성
    seq_volt = np.zeros((1, length))
    seq_capa = np.zeros((1, length))

    # 시퀀스 데이터를 만들 대상 피처의 데이터프레임에서의 인덱스
    # 사용하는 데이터프레임에 따라 인덱스 변경 필요
    f1_index = 1
    f2_index = 2

    # 시퀀스 길이만큼 반복하면 값 추출 후 저장
    for i in range(1, length + 1) :
        seq_volt[:, i-1] = df.iloc[-i, f1_index] # 전압의 경우 현재 전압 포함
        seq_capa[:, i-1] = df.iloc[-i - 1, f2_index] # 투입용량의 경우 현재는 고려하지 않음

    # 스케일링 수행
    seq_volt = (seq_volt - v_min) / (v_max - v_min)
    seq_capa = (seq_capa - c_min) / (c_max - c_min)

    # 이전의 데이터가 먼저 나오도록 배열을 반대로 뒤집어줌
    res[:,:,0] = np.flip(seq_volt)
    res[:,:,1] = np.flip(seq_capa)

    # 결과 반환
    return res


# 최적화 수행 함수
def optimize(cap_pred, params_now, n_used) :

    # 가변형 리액터 탭별 투입용량
    tap_cap = [0, 97, 101, 105, 109, 114, 119, 124, 129, 135, 140, 147, 153, 160, 167, 175, 183, 192, 200]

    # 결과 저장 테이블
    table = np.zeros((608, 8))

    # 현재 전력설비 상태 저장
    x1_now, x2_now, x3_now, x4_now, x5_now, tap_now = params_now

    # Cost 및 투입용량 계산 후 저장
    cnt = 0
    for x1 in range(2) :
        for x2 in range(2) :
            for x3 in range(2) :
                for x4 in range(2) :
                    for x5 in range(2) :
                        for tap in range(0, 19) :
                            # 투입용량
                            cap = 200 * (x1 + x2 + x3 + x4 + x5) + tap_cap[tap]

                            # Cost
                            cost = (n_used[0] * np.abs(x1 - x1_now) + n_used[1] * np.abs(x2 - x2_now)
                            + n_used[2] * np.abs(x3 - x3_now) + n_used[3] * np.abs(x4 - x4_now)
                            + n_used[4] * np.abs(x5 - x5_now) + (n_used[5] / 18) * np.abs(tap - tap_now))

                            # 계산 결과 저장
                            table[cnt] = [x1, x2, x3, x4, x5, tap, cap, cost]

                            cnt += 1

    # 제약식 1
    cons1_ind = np.where(table[:,6] >= cap_pred)
    table = table[cons1_ind]

    # 제약식 2
    cons2_ind = np.where(table[:,6] - cap_pred <= 10)

    if len(cons2_ind[0]) == 0 :

        cons2_ind = np.where(table[:,6] - cap_pred <= 96)

        table = table[cons2_ind]

    else :

        table = table[cons2_ind]

    # 최종 결과
    optimized_ind = np.argmin(table[:,-1])

    # 결과 저장
    x1_target, x2_target, x3_target, x4_target, x5_target, tap_target = table[optimized_ind,:6]

    # 결과 반환
    # 일반 리액터 1, 2, 3, 4, 5 조정 방안 (0 or 1) / 가변형 리택터 1 조정 방안 (0 ~ 18), 최적화 진행 후 투입 용량
    return x1_target, x2_target, x3_target, x4_target, x5_target, tap_target, table[optimized_ind,6]
