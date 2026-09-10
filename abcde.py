"""BACI 표본 데이터를 위한 Streamlit 무역 분석 대시보드."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="무역 분석 대시보드", page_icon="📦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """대소문자/공백 차이를 무시하고 첫 번째 일치 열을 돌려준다."""
    normalized = {str(col).strip().lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


@st.cache_data(show_spinner=False)
def load_data(trade_file: Path, country_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(trade_file), pd.read_csv(country_file)


def make_country_lookup(countries: pd.DataFrame) -> dict:
    code_col = find_column(countries, ["country_code", "code", "country_id", "i"])
    name_col = find_column(countries, ["country_name", "country", "name", "country_label"])
    if code_col is None or name_col is None:
        return {}
    return dict(zip(countries[code_col], countries[name_col]))


def to_display_name(code, lookup: dict) -> str:
    name = lookup.get(code)
    return str(name) if pd.notna(name) else str(code)


def main() -> None:
    st.title("무역 분석 대시보드")

    default_trade = BASE_DIR / "baci_85_sample.csv"
    default_country = BASE_DIR / "country_code_sample.csv"
    if not default_trade.exists() or not default_country.exists():
        st.warning("같은 폴더에 `baci_85_sample.csv`와 `country_code_sample.csv`를 넣어 주세요.")
        st.caption("두 파일을 찾으면 대시보드가 자동으로 표시됩니다.")
        return

    try:
        trade, countries = load_data(default_trade, default_country)
    except Exception as error:
        st.error(f"CSV 파일을 읽을 수 없습니다: {error}")
        return

    year_col = find_column(trade, ["t", "year", "year_id"])
    exporter_col = find_column(trade, ["i", "exporter", "exporter_code", "country_code"])
    value_col = find_column(trade, ["v", "value", "export_value", "trade_value"])
    if not all([year_col, exporter_col, value_col]):
        st.error("무역 파일에는 연도(t), 수출국(i), 수출액(v) 열이 필요합니다.")
        st.code("현재 열: " + ", ".join(map(str, trade.columns)))
        return

    # 수출액을 수치로 변환한 뒤, 전체 분포의 3분위수를 기준으로 대·중·소를 정한다.
    data = trade.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    q33, q67 = data[value_col].quantile([1 / 3, 2 / 3])
    data["무역액 등급"] = pd.cut(
        data[value_col],
        bins=[float("-inf"), q33, q67, float("inf")],
        labels=["소", "중", "대"],
        include_lowest=True,
    )

    lookup = make_country_lookup(countries)
    data["국가"] = data[exporter_col].map(lambda x: to_display_name(x, lookup))

    st.sidebar.header("필터")
    country_options = sorted(data["국가"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect("국가 선택", country_options, default=country_options)
    selected_grades = st.sidebar.multiselect("무역액 등급 선택", ["대", "중", "소"], default=["대", "중", "소"])

    filtered = data[
        data["국가"].isin(selected_countries) & data["무역액 등급"].isin(selected_grades)
    ].copy()

    st.subheader("결측치 현황")
    missing = trade.isna().sum().rename("결측치 수").to_frame()
    missing["결측 비율(%)"] = (missing["결측치 수"] / len(trade) * 100).round(2) if len(trade) else 0
    st.dataframe(missing, use_container_width=True)

    total_count = len(filtered)
    total_value = filtered[value_col].sum(min_count=1)
    left, right = st.columns(2)
    left.metric("총 거래건수", f"{total_count:,}건")
    right.metric("총 수출액(달러)", f"${(total_value if pd.notna(total_value) else 0):,.2f}")

    if filtered.empty:
        st.info("선택한 조건에 맞는 데이터가 없습니다.")
        return

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("국가 × 연도 수출액 히트맵 (상위 8개국)")
        top_countries = (
            filtered.groupby("국가", as_index=False)[value_col].sum().nlargest(8, value_col)["국가"]
        )
        matrix = filtered[filtered["국가"].isin(top_countries)].pivot_table(
            index="국가", columns=year_col, values=value_col, aggfunc="sum", fill_value=0
        )
        matrix = matrix.reindex(top_countries)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=.4, ax=ax)
        ax.set_xlabel("연도")
        ax.set_ylabel("국가")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with chart_right:
        st.subheader("무역액 등급분포")
        distribution = filtered["무역액 등급"].value_counts().reindex(["대", "중", "소"], fill_value=0)
        st.bar_chart(distribution, color="#3979b8")
        st.caption("수출액 전체 분포의 3분위수 기준: 대·중·소")

    st.subheader("상위 5개국 × 무역액 등급 교차표")
    top5 = filtered.groupby("국가")[value_col].sum().nlargest(5).index
    crosstab = pd.crosstab(filtered.loc[filtered["국가"].isin(top5), "국가"], filtered.loc[filtered["국가"].isin(top5), "무역액 등급"])
    crosstab = crosstab.reindex(columns=["대", "중", "소"], fill_value=0)
    raw_tab, normalized_tab = st.tabs(["원본 건수", "정규화 비율"])
    with raw_tab:
        st.dataframe(crosstab, use_container_width=True)
    with normalized_tab:
        st.dataframe((crosstab.div(crosstab.sum(axis=1), axis=0).fillna(0) * 100).round(2).astype(str) + "%", use_container_width=True)


if __name__ == "__main__":
    main()
