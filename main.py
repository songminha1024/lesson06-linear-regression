
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================
# 1. 페이지 설정
# =====================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온 데이터를 분석하고, "
    "선형회귀를 이용해 선택한 연도의 예상 기온을 계산합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =====================================
# 2. 데이터 불러오기 및 전처리
# =====================================
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

    # 2025년까지의 자료만 사용
    df = df[df["날짜"].dt.year <= 2025].copy()
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일수가 300일 미만인 연도 제외
    yearly = yearly[yearly["관측일수"] >= 300].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.info("인터넷 연결과 CSV 파일의 주소를 확인해 주세요.")
    st.stop()


# =====================================
# 3. 회귀 분석
# =====================================
if len(yearly) < 2:
    st.error("회귀 분석에 사용할 수 있는 연도가 2개 미만입니다.")
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# 전체 기간의 선형회귀
slope, intercept = np.polyfit(years, temps, 1)

# 피어슨 상관계수
correlation = np.corrcoef(years, temps)[0, 1]

start_year = int(years.min())
end_year = int(years.max())

# 100년당 기온 변화량
slope_per_century = slope * 100

# =====================================
# 4. 최근 20년의 선형회귀
# =====================================
recent_start = end_year - 19

recent = yearly[
    (yearly["연도"] >= recent_start) &
    (yearly["연도"] <= end_year)
].copy()

recent_available = (
    len(recent) >= 2 and recent["연도"].nunique() >= 2
)

if recent_available:
    recent_years = recent["연도"].to_numpy(dtype=float)
    recent_temps = recent["연평균기온"].to_numpy(dtype=float)

    recent_slope, recent_intercept = np.polyfit(
        recent_years, recent_temps, 1
    )

    recent_slope_per_century = recent_slope * 100
else:
    recent_slope = None
    recent_intercept = None
    recent_slope_per_century = None


# =====================================
# 5. 연도 선택 및 예상 기온
# =====================================
st.subheader("🔮 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C"
)

st.caption(
    "예상 기온은 전체 분석 기간의 선형회귀 직선을 연장해 계산합니다. "
    "실제 미래 기온이나 공식 기후 예보를 의미하지는 않습니다."
)


# =====================================
# 6. 100년당 기온 변화량 비교
# =====================================
st.subheader("📈 100년당 기온 변화량 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year}년)",
        value=f"{slope_per_century:+.2f} °C / 100년"
    )

with col2:
    if recent_available:
        difference = (
            recent_slope_per_century - slope_per_century
        )

        st.metric(
            label=f"최근 20년 ({recent_start}~{end_year}년)",
            value=f"{recent_slope_per_century:+.2f} °C / 100년",
            delta=f"{difference:+.2f} °C / 100년 (전체 대비)"
        )
    else:
        st.metric(
            label=f"최근 20년 ({recent_start}~{end_year}년)",
            value="계산 불가"
        )

st.caption(
    "양수는 기온 상승 추세, 음수는 하강 추세를 뜻합니다. "
    "최근 20년 분석에는 관측일수 조건을 통과한 연도만 사용합니다."
)


# =====================================
# 7. 회귀 분석 정보
# =====================================
st.subheader("📊 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 분석 연도 수", f"{len(yearly)}년")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.4f}")

st.write(
    f"**전체 기간 회귀식:** "
    f"연평균기온 = {slope:.5f} × 연도 + ({intercept:.3f})"
)

if recent_available:
    st.write(
        f"**최근 20년 회귀식:** "
        f"연평균기온 = {recent_slope:.5f} × 연도 "
        f"+ ({recent_intercept:.3f})"
    )


# =====================================
# 8. Plotly 그래프
# =====================================
st.subheader("연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온 산점도
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

# 전체 기간 회귀 직선
line_years = np.linspace(start_year, end_year, 300)
line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 최근 20년 회귀 직선
if recent_available:
    recent_line_years = np.linspace(
        recent["연도"].min(),
        recent["연도"].max(),
        100
    )

    recent_line_temps = (
        recent_slope * recent_line_years + recent_intercept
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_years,
            y=recent_line_temps,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, dash="dash"),
            hovertemplate=(
                "연도: %{x:.0f}년<br>"
                "최근 20년 회귀 기온: %{y:.2f} °C"
                "<extra></extra>"
            )
        )
    )

# 선택한 연도의 예상 기온
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name="선택한 연도의 예상 기온",
        marker=dict(size=14, symbol="star"),
        hovertemplate=(
            "예측 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연평균기온의 변화 추세",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    template="plotly_white",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig, use_container_width=True)


# =====================================
# 9. 연도별 데이터 표
# =====================================
st.subheader("연도별 분석 데이터")

display_df = yearly.rename(
    columns={
        "연도": "연도",
        "연평균기온": "연평균기온 (°C)",
        "관측일수": "관측일수 (일)"
    }
).copy()

display_df["연평균기온 (°C)"] = (
    display_df["연평균기온 (°C)"].round(2)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "데이터 출처: greatsong/modudata 서울 기온 CSV. "
    "연평균기온은 유효한 일별 평균기온의 산술평균입니다."
)
