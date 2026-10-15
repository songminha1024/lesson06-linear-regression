
# 회귀 분석에 사용할 데이터
if len(yearly) < 2:
    st.error("회귀 분석을 하려면 조건을 만족하는 연도가 2개 이상 필요합니다.")
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# 전체 기간 선형회귀
slope, intercept = np.polyfit(years, temps, 1)
correlation = np.corrcoef(years, temps)[0, 1]

start_year = int(years.min())
end_year = int(years.max())

# 기울기를 100년당 기온 변화량으로 환산
slope_per_century = slope * 100

# 최근 20년: 가장 최근 연도를 포함한 20년 구간
recent_start = end_year - 19
recent = yearly[
    (yearly["연도"] >= recent_start) &
    (yearly["연도"] <= end_year)
]

if len(recent) >= 2 and recent["연도"].nunique() >= 2:
    recent_slope, recent_intercept = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1
    )
    recent_slope_per_century = recent_slope * 100
    recent_available = True
else:
    recent_slope_per_century = None
    recent_available = False

# 슬라이더
