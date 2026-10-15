
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 기본 설정
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.caption(
    "서울의 연평균기온 데이터를 이용해 "
    "연도별 기온 추세를 분석합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"필수 열이 없습니다: {', '.join(sorted(missing))}"
        )

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 수업의 기준 기간: 2025년까지
    df = df[df["연도"] <= 2025]

    # 날짜별 유효한 평균기온 관측 건수와 연평균기온 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 연도 제외
    yearly = yearly[yearly["관측일수"] >= 300]
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.info("인터넷 연결과 데이터 파일의 열 이름을 확인해 주세요.")
    st.stop()

# 회귀 분석에 사용할 데이터
if len(yearly) < 2:
    st.error("회귀 분석을 하려면 조건을 만족하는 연도가 2개 이상 필요합니다.")
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# 선형회귀: 기온 = 기울기 * 연도 + 절편
slope, intercept = np.polyfit(years, temps, 1)

# 피어슨 상관계수
correlation = np.corrcoef(years, temps)[0, 1]

start_year = int(years.min())
end_year = int(years.max())

# 슬라이더
st.subheader("원하는 연도의 예상 평균기온")

selected_year = st.slider(
    "예측 연도",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

predicted_temp = slope * selected_year + intercept

# 예상 기온 크게 표시
st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C"
)

st.caption(
    "예상값은 분석 기간의 선형 추세를 연장한 계산값입니다. "
    "실제 기온이나 공식 기후 예보와는 다를 수 있습니다."
)

# 분석 요약
st.subheader("회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{len(yearly)}년")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.4f}")

st.write(
    f"**회귀식:** 연평균기온 = "
    f"{slope:.5f} × 연도 + ({intercept:.3f})"
)

# 산점도와 회귀 직선
fig = go.Figure()

# 실제 관측값
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선은 실제 분석 기간에만 표시
line_years = np.linspace(start_year, end_year, 300)
line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도의 예상값 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name="선택한 연도의 예상값",
        marker=dict(size=13, symbol="star"),
        hovertemplate=(
            "예측 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연평균기온과 선형회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("연도별 분석 데이터")
st.dataframe(
    yearly.rename(
        columns={
            "연도": "연도",
            "연평균기온": "연평균기온 (°C)",
            "관측일수": "관측일수 (일)"
        }
    ).style.format({"연평균기온 (°C)": "{:.2f}"}),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "데이터 출처: "
    "greatsong/modudata의 서울 기온 CSV. "
    "연평균기온은 유효한 일별 평균기온의 산술평균입니다."
)
