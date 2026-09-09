"""BACI 표본 CSV를 시각화하는 Streamlit 무역 분석 대시보드."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="무역 분석 대시보드", page_icon="📊", layout="wide")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parent
TRADE_PATH = DATA_DIR / "baci_85_sample.csv"
COUNTRY_PATH = DATA_DIR / "country_codes_sample.csv"


@st.cache_data
def load_trade_data() -> pd.DataFrame:
    """BACI 데이터와 국가코드 데이터를 결합하고 분석용 열을 만든다."""
    trade = pd.read_csv(TRADE_PATH)
    countries = pd.read_csv(COUNTRY_PATH)

    # BACI: i=수출국, j=수입국(상대국), k=HS 품목코드, t=연도, v=수출액(천 달러)
    data = trade.merge(countries, on="j", how="left")
    data["country_name"] = data["country_name"].fillna("국가코드 " + data["j"].astype(str))
    data["v"] = pd.to_numeric(data["v"], errors="coerce")
    low, high = data["v"].quantile([1 / 3, 2 / 3])
    data["무역액 등급"] = pd.cut(
        data["v"],
        bins=[float("-inf"), low, high, float("inf")],
        labels=["소", "중", "대"],
        include_lowest=True,
    )
    return data


def main() -> None:
    st.title("무역 분석 대시보드")
    st.caption("BACI 수출 표본 데이터 · 수출액 단위: 천 달러")

    if not TRADE_PATH.exists() or not COUNTRY_PATH.exists():
        st.error("`baci_85_sample.csv`와 `country_codes_sample.csv`를 abcd.py와 같은 폴더에 넣어 주세요.")
        return

    try:
        data = load_trade_data()
    except (OSError, KeyError, pd.errors.ParserError) as error:
        st.error(f"CSV를 불러오는 중 오류가 발생했습니다: {error}")
        return

    st.sidebar.header("필터")
    all_countries = sorted(data["country_name"].unique())
    selected_countries = st.sidebar.multiselect("국가 선택", all_countries, default=all_countries)
    selected_grades = st.sidebar.multiselect("무역액 등급 선택", ["대", "중", "소"], default=["대", "중", "소"])
    filtered = data[
        data["country_name"].isin(selected_countries)
        & data["무역액 등급"].isin(selected_grades)
    ].copy()

    st.subheader("BACI 원본 파일 결측치")
    missing = data[["i", "j", "k", "t", "v"]].isna().sum().rename("결측치 수").to_frame()
    missing["결측 비율(%)"] = (missing["결측치 수"] / len(data) * 100).round(2)
    st.dataframe(missing, width="stretch")

    metric_a, metric_b = st.columns(2)
    metric_a.metric("총 거래건수", f"{len(filtered):,}건")
    metric_b.metric("총 수출액(달러)", f"${filtered['v'].sum() * 1_000:,.0f}")
    if filtered.empty:
        st.info("선택한 필터에 해당하는 거래 데이터가 없습니다.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("국가 × 연도 수출액 히트맵 (상위 8개국)")
        top8 = filtered.groupby("country_name")["v"].sum().nlargest(8).index
        heatmap = filtered[filtered["country_name"].isin(top8)].pivot_table(
            index="country_name", columns="t", values="v", aggfunc="sum", fill_value=0
        ).reindex(top8)
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.heatmap(heatmap, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.5, ax=axis)
        axis.set_xlabel("연도")
        axis.set_ylabel("국가")
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    with right:
        st.subheader("무역액 등급분포")
        grade_counts = filtered["무역액 등급"].value_counts().reindex(["대", "중", "소"], fill_value=0)
        st.bar_chart(grade_counts, color="#2563eb")
        st.caption("전체 수출액의 3분위수 기준으로 대·중·소를 구분했습니다.")

    st.subheader("상위 5개국 × 무역액 등급 교차표")
    top5 = filtered.groupby("country_name")["v"].sum().nlargest(5).index
    top5_data = filtered[filtered["country_name"].isin(top5)]
    table = pd.crosstab(top5_data["country_name"], top5_data["무역액 등급"]).reindex(
        index=top5, columns=["대", "중", "소"], fill_value=0
    )
    raw_tab, normalized_tab = st.tabs(["원본 건수", "정규화 비율"])
    with raw_tab:
        st.dataframe(table, width="stretch")
    with normalized_tab:
        normalized = (table.div(table.sum(axis=1), axis=0).fillna(0) * 100).round(2)
        st.dataframe(normalized.style.format("{:.2f}%"), width="stretch")


if __name__ == "__main__":
    main()
