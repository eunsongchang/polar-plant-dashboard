import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import unicodedata
from pathlib import Path
import io

# --- 1. Streamlit 설정 및 한글 폰트 CSS 주입 ---
st.set_page_config(
    page_title="🌱 극지식물 최적 EC 농도 연구",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 깨짐 방지 CSS (Noto Sans KR 적용)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Plotly 기본 폰트 설정 (그래프 내 한글 깨짐 방지)
PLOTLY_FONT = "Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif"

# --- 2. 상수 정의 ---
# 학교별 EC 목표값 및 색상 매핑
EC_MAPPING = {
    "송도고": {"EC_goal": 1.0, "color": "#1f77b4"},
    "하늘고": {"EC_goal": 2.0, "color": "#2ca02c"}, # 최적 EC 강조 색상
    "아라고": {"EC_goal": 4.0, "color": "#ff7f0e"},
    "동산고": {"EC_goal": 8.0, "color": "#d62728"},
}

# --- 3. 데이터 로딩 및 전처리 (@st.cache_data 사용) ---

@st.cache_data(show_spinner="🗂️ 환경 데이터를 로드하고 전처리 중...")
def load_and_preprocess_env_data(data_dir: Path):
    """
    환경 데이터 (CSV)를 로드하고 전처리합니다.
    한글 파일명(NFC/NFD) 문제를 방지하기 위해 unicodedata를 사용합니다.
    """
    all_env_data = []
    
    # 1. 파일 목록 찾기 (pathlib.Path.iterdir() 사용)
    for path in data_dir.iterdir():
        if path.is_file() and path.suffix.lower() == '.csv':
            # 2. NFC/NFD 문제 방지를 위한 정규화 및 학교명 추출
            normalized_name = unicodedata.normalize("NFC", path.name)
            
            # 파일명에서 '고'로 끝나고, '환경데이터.csv'로 끝나는 학교명을 식별
            if '환경데이터' in normalized_name:
                try:
                    # '송도고_환경데이터.csv' -> '송도고' 추출
                    school_name = normalized_name.split('_')[0] 
                    if school_name not in EC_MAPPING: continue # 매핑되지 않은 파일은 스킵

                    df = pd.read_csv(path, encoding='utf-8')
                    
                    # 컬럼명 정리 및 데이터 타입 변환
                    df.columns = ['time', 'temperature', 'humidity', 'ph', 'ec']
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    df.dropna(subset=['time'], inplace=True)
                    
                    # 숫자형 컬럼 변환 (결측치 처리)
                    for col in ['temperature', 'humidity', 'ph', 'ec']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df['school'] = school_name
                    df['ec_goal'] = EC_MAPPING[school_name]['EC_goal']
                    all_env_data.append(df)
                    
                except Exception as e:
                    st.error(f"환경 데이터 파일 로드 오류 ({normalized_name}): {e}")
                    
    if not all_env_data:
        st.error("데이터 디렉토리에서 유효한 환경 데이터(CSV) 파일을 찾을 수 없습니다.")
        return pd.DataFrame(), pd.DataFrame()

    env_df = pd.concat(all_env_data, ignore_index=True)
    
    # 통계 요약 (학교별 평균)
    env_summary_df = env_df.groupby('school').agg(
        avg_temp=('temperature', 'mean'),
        avg_humidity=('humidity', 'mean'),
        avg_ph=('ph', 'mean'),
        avg_ec=('ec', 'mean'),
        ec_goal=('ec_goal', 'first'),
        count=('time', 'count')
    ).reset_index()
    
    return env_df, env_summary_df

@st.cache_data(show_spinner="🔬 생육 데이터를 로드하고 전처리 중...")
def load_and_preprocess_growth_data(data_dir: Path):
    """
    생육 결과 데이터 (XLSX)를 로드하고 전처리합니다.
    시트 이름 하드코딩을 피하고, NFC/NFD 문제에 대비합니다.
    """
    all_growth_data = []
    
    # 1. 파일 목록 찾기 (pathlib.Path.iterdir() 사용)
    xlsx_path = None
    for path in data_dir.iterdir():
        if path.is_file() and path.suffix.lower() == '.xlsx':
             # 파일명에서 '생육결과데이터'를 포함하는 파일을 찾음
            normalized_name = unicodedata.normalize("NFC", path.name)
            if '생육결과데이터' in normalized_name:
                xlsx_path = path
                break

    if not xlsx_path:
        st.error("데이터 디렉토리에서 유효한 생육 결과 데이터(XLSX) 파일을 찾을 수 없습니다.")
        return pd.DataFrame(), pd.DataFrame()

    # 2. 시트 이름 동적 로드 (하드코딩 방지)
    try:
        xls = pd.ExcelFile(xlsx_path, engine='openpyxl')
        sheet_names = [unicodedata.normalize("NFC", name) for name in xls.sheet_names]
    except Exception as e:
        st.error(f"생육 결과 파일 읽기 오류: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # 3. 각 시트 로드 및 전처리
    for sheet_name in sheet_names:
        if sheet_name in EC_MAPPING: # 시트 이름이 학교명인 경우만 처리
            try:
                df = xls.parse(sheet_name)
                
                # 컬럼명 정리
                df.columns = ['individual_id', 'leaf_count', 'shoot_length', 'root_length', 'fresh_weight']
                
                # 숫자형 컬럼 변환
                for col in ['leaf_count', 'shoot_length', 'root_length', 'fresh_weight']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['school'] = sheet_name
                df['ec_goal'] = EC_MAPPING[sheet_name]['EC_goal']
                all_growth_data.append(df)
            except Exception as e:
                st.warning(f"생육 결과 시트 로드 오류 ({sheet_name}): {e}")
                
    if not all_growth_data:
        st.error("생육 결과 파일에서 유효한 학교 시트를 찾을 수 없습니다.")
        return pd.DataFrame(), pd.DataFrame()
        
    growth_df = pd.concat(all_growth_data, ignore_index=True)
    
    # 통계 요약 (EC별 평균)
    growth_summary_df = growth_df.groupby('ec_goal').agg(
        avg_fresh_weight=('fresh_weight', 'mean'),
        avg_leaf_count=('leaf_count', 'mean'),
        avg_shoot_length=('shoot_length', 'mean'),
        count=('individual_id', 'count')
    ).reset_index()
    growth_summary_df['ec_goal'] = growth_summary_df['ec_goal'].astype(str) + ' EC'

    return growth_df, growth_summary_df


# --- 4. 데이터 로드 실행 ---
DATA_DIR = Path("./data")

if not DATA_DIR.exists():
    st.error(f"⚠️ 데이터 디렉토리({DATA_DIR.resolve()})를 찾을 수 없습니다. 파일 구조를 확인해 주세요.")
    st.stop()
    
# 데이터 로드
env_df, env_summary_df = load_and_preprocess_env_data(DATA_DIR)
growth_df, growth_summary_df = load_and_preprocess_growth_data(DATA_DIR)

if env_df.empty and growth_df.empty:
    st.error("⚠️ 모든 데이터 로드에 실패했습니다. 파일명, 인코딩, 파일 구조를 확인해 주세요.")
    st.stop()


# --- 5. 사이드바 및 필터 ---
school_options = ["전체"] + list(EC_MAPPING.keys())
selected_school = st.sidebar.selectbox(
    "🏫 학교 선택",
    school_options,
    index=1 # 기본값: 송도고
)

# 필터링
if selected_school != "전체":
    filtered_env_df = env_df[env_df['school'] == selected_school].copy()
    filtered_growth_df = growth_df[growth_df['school'] == selected_school].copy()
else:
    filtered_env_df = env_df.copy()
    filtered_growth_df = growth_df.copy()

# --- 6. 앱 본문 레이아웃 ---
st.title("🌱 극지식물 최적 EC 농도 연구")
st.caption("Streamlit Cloud 환경 최적화, 한글 파일/폰트 깨짐 완벽 방지 대시보드")

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ==============================================================================
# Tab 1: 실험 개요
# ==============================================================================
with tab1:
    st.header("연구 배경 및 목적")
    st.markdown("""
    * **연구 배경:** 극한 환경에 적응하는 극지식물의 생육 특성 분석을 위해, 수경재배 환경에서 필수적인 **EC(전기 전도도)** 농도를 학교별로 다르게 설정하여 그 영향을 비교합니다.
    * **연구 목적:** 다양한 EC 농도(1.0, 2.0, 4.0, 8.0) 조건 하에서 극지식물의 생육 결과를 비교하고, **최적 생육을 유도하는 EC 농도**를 도출하는 것입니다.
    """)
    
    st.subheader("학교별 EC 조건 및 개체수")
    
    # 학교별 EC 조건 표
    ec_data = []
    for school, data in EC_MAPPING.items():
        growth_count = growth_df[growth_df['school'] == school].shape[0] if not growth_df.empty else 0
        ec_data.append({
            "학교명": school,
            "EC 목표 (mS/cm)": data["EC_goal"],
            "색상": data["color"],
            "총 개체수": f"{growth_count}개체"
        })
    ec_table = pd.DataFrame(ec_data)
    st.dataframe(ec_table, 
                 hide_index=True, 
                 column_config={"색상": st.column_config.Color(width="small")},
                 use_container_width=True)

    st.subheader("주요 지표 요약")
    
    # 주요 지표 계산
    total_individuals = growth_df.shape[0] if not growth_df.empty else 0
    
    if not env_df.empty:
        avg_temp_all = env_df['temperature'].mean()
        avg_humidity_all = env_df['humidity'].mean()
    else:
        avg_temp_all = np.nan
        avg_humidity_all = np.nan

    # 최적 EC 도출 (생중량 기준)
    if not growth_summary_df.empty:
        best_ec_row = growth_summary_df.loc[growth_summary_df['avg_fresh_weight'].idxmax()]
        best_ec = f"{best_ec_row['ec_goal'].split(' ')[0]} mS/cm"
        best_weight = f"{best_ec_row['avg_fresh_weight']:.2f} g"
    else:
        best_ec = "N/A"
        best_weight = "N/A"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="총 개체수", value=f"{total_individuals} 개체")
    with col2:
        st.metric(label="전체 평균 온도", value=f"{avg_temp_all:.1f} °C" if not np.isnan(avg_temp_all) else "N/A")
    with col3:
        st.metric(label="전체 평균 습도", value=f"{avg_humidity_all:.1f} %" if not np.isnan(avg_humidity_all) else "N/A")
    with col4:
        st.metric(label="🏆 최적 EC (평균 생중량)", value=best_ec, delta=f"평균 {best_weight}", delta_color="normal")


# ==============================================================================
# Tab 2: 환경 데이터
# ==============================================================================
with tab2:
    if env_summary_df.empty:
        st.error("환경 데이터가 존재하지 않아 그래프를 표시할 수 없습니다.")
    else:
        st.header("학교별 환경 평균 비교")
        
        # 2x2 서브플롯 생성
        fig_env_comp = make_subplots(rows=2, cols=2, 
                                     subplot_titles=("평균 온도 (°C)", "평균 습도 (%)", 
                                                     "평균 pH", "목표 EC vs 실측 EC (mS/cm)"))

        # 1. 평균 온도
        fig_env_comp.add_trace(go.Bar(x=env_summary_df['school'], y=env_summary_df['avg_temp'], name='온도',
                                      marker_color=[EC_MAPPING[s]['color'] for s in env_summary_df['school']]), row=1, col=1)

        # 2. 평균 습도
        fig_env_comp.add_trace(go.Bar(x=env_summary_df['school'], y=env_summary_df['avg_humidity'], name='습도',
                                      marker_color=[EC_MAPPING[s]['color'] for s in env_summary_df['school']]), row=1, col=2)

        # 3. 평균 pH
        fig_env_comp.add_trace(go.Bar(x=env_summary_df['school'], y=env_summary_df['avg_ph'], name='pH',
                                      marker_color=[EC_MAPPING[s]['color'] for s in env_summary_df['school']]), row=2, col=1)

        # 4. 목표 EC vs 실측 EC (이중 막대)
        fig_env_comp.add_trace(go.Bar(x=env_summary_df['school'], y=env_summary_df['ec_goal'], name='목표 EC',
                                      marker_color='gray', opacity=0.6), row=2, col=2)
        fig_env_comp.add_trace(go.Bar(x=env_summary_df['school'], y=env_summary_df['avg_ec'], name='실측 EC',
                                      marker_color=[EC_MAPPING[s]['color'] for s in env_summary_df['school']]), row=2, col=2)
        
        # 레이아웃 설정
        fig_env_comp.update_layout(height=700, showlegend=False, 
                                   font=dict(family=PLOTLY_FONT),
                                   title_text="**학교별 환경 인자 평균 비교**")
        fig_env_comp.update_xaxes(title_text="학교명")
        
        st.plotly_chart(fig_env_comp, use_container_width=True)

        st.markdown("---")

        # 학교별 시계열 데이터
        if selected_school != "전체" and not filtered_env_df.empty:
            st.header(f"📈 {selected_school} 환경 변화 (시계열)")

            col_ts_1, col_ts_2, col_ts_3 = st.columns(3)
            
            # 1. 온도 변화
            with col_ts_1:
                fig_temp = px.line(filtered_env_df, x='time', y='temperature', 
                                   title='온도 변화', labels={'temperature': '온도 (°C)', 'time': '시간'},
                                   color_discrete_sequence=[EC_MAPPING[selected_school]['color']])
                fig_temp.update_layout(font=dict(family=PLOTLY_FONT))
                st.plotly_chart(fig_temp, use_container_width=True)

            # 2. 습도 변화
            with col_ts_2:
                fig_humidity = px.line(filtered_env_df, x='time', y='humidity', 
                                       title='습도 변화', labels={'humidity': '습도 (%)', 'time': '시간'},
                                       color_discrete_sequence=[EC_MAPPING[selected_school]['color']])
                fig_humidity.update_layout(font=dict(family=PLOTLY_FONT))
                st.plotly_chart(fig_humidity, use_container_width=True)

            # 3. EC 변화 (목표 EC 수평선 추가)
            with col_ts_3:
                fig_ec = px.line(filtered_env_df, x='time', y='ec', 
                                  title='EC 변화', labels={'ec': 'EC (mS/cm)', 'time': '시간'},
                                  color_discrete_sequence=[EC_MAPPING[selected_school]['color']])
                
                # 목표 EC 수평선 추가
                ec_goal_val = EC_MAPPING[selected_school]['EC_goal']
                fig_ec.add_hline(y=ec_goal_val, line_dash="dash", line_color="gray", 
                                 annotation_text=f"목표 EC: {ec_goal_val}", 
                                 annotation_position="bottom right",
                                 annotation=dict(font=dict(family=PLOTLY_FONT)))
                                 
                fig_ec.update_layout(font=dict(family=PLOTLY_FONT))
                st.plotly_chart(fig_ec, use_container_width=True)

        elif selected_school == "전체":
             st.info("개별 학교의 시계열 변화를 보려면 사이드바에서 학교를 선택하세요.")

        # 환경 데이터 원본
        with st.expander("원본 환경 데이터 테이블 및 다운로드"):
            st.dataframe(filtered_env_df.drop(columns=['ec_goal'], errors='ignore'), use_container_width=True)
            
            # CSV 다운로드
            if not filtered_env_df.empty:
                @st.cache_data
                def convert_df_to_csv(df):
                    # BOM 추가하여 엑셀에서 한글 깨짐 방지
                    return df.to_csv(index=False, encoding='utf-8-sig')
                
                csv = convert_df_to_csv(filtered_env_df)
                st.download_button(
                    label="다운로드: 환경 데이터 (CSV)",
                    data=csv,
                    file_name=f"{selected_school}_환경데이터_raw.csv",
                    mime="text/csv",
                )


# ==============================================================================
# Tab 3: 생육 결과
# ==============================================================================
with tab3:
    if growth_df.empty:
        st.error("생육 결과 데이터가 존재하지 않아 분석을 표시할 수 없습니다.")
    else:
        st.header("EC별 생육 결과 비교 분석")
        
        # 1. 핵심 결과 카드: EC별 평균 생중량 (최댓값 강조)
        if not growth_summary_df.empty:
            st.subheader("🥇 핵심 지표: EC별 평균 생중량")
            
            best_ec_val = growth_summary_df['avg_fresh_weight'].max()
            best_ec_label = growth_summary_df[growth_summary_df['avg_fresh_weight'] == best_ec_val]['ec_goal'].iloc[0]
            
            cols = st.columns(growth_summary_df.shape[0])
            for i, row in growth_summary_df.iterrows():
                school_name = [s for s, data in EC_MAPPING.items() if data['EC_goal'] == float(row['ec_goal'].split(' ')[0])][0]
                color = EC_MAPPING[school_name]['color']
                
                delta_val = f"총 {row['count']} 개체"
                
                # 최적값 강조
                if row['ec_goal'] == best_ec_label:
                    st.markdown(f"""
                    <div style='background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                        <p style='font-size: 14px; margin-bottom: 0;'>🏆 {row['ec_goal']} ({school_name})</p>
                        <h3 style='margin-top: 5px; margin-bottom: 0;'>{row['avg_fresh_weight']:.2f} g</h3>
                        <p style='font-size: 12px; margin-top: 0;'>{delta_val}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with cols[i]:
                        st.metric(label=f"{row['ec_goal']} ({school_name})", 
                                  value=f"{row['avg_fresh_weight']:.2f} g", 
                                  delta=delta_val, 
                                  delta_color="off")
        
            st.markdown("---")


        # 2. EC별 생육 비교 (2x2 막대 그래프)
        st.subheader("EC별 생육 지표 평균 비교")
        fig_growth_comp = make_subplots(rows=2, cols=2, 
                                        subplot_titles=("평균 생중량 (g) ⭐", "평균 잎 수 (장)", 
                                                        "평균 지상부 길이 (mm)", "개체수"))

        # Plotting Helper
        def add_bar_trace(df, y_col, name, row, col):
            # EC 목표값에 따른 색상 매핑
            colors = [EC_MAPPING[[s for s, d in EC_MAPPING.items() if d['EC_goal'] == float(ec.split(' ')[0])][0]]['color'] 
                      for ec in df['ec_goal']]
            
            # 가장 중요한 생중량 그래프에 최댓값 표시
            if y_col == 'avg_fresh_weight':
                max_val = df[y_col].max()
                text_values = [f"{val:.2f}" + (" (MAX)" if val == max_val else "") for val in df[y_col]]
            elif y_col == 'count':
                text_values = [str(val) for val in df[y_col]]
            else:
                text_values = [f"{val:.1f}" for val in df[y_col]]

            fig_growth_comp.add_trace(go.Bar(x=df['ec_goal'], y=df[y_col], name=name,
                                              marker_color=colors, text=text_values, 
                                              textposition='outside'), row=row, col=col)

        # 1. 평균 생중량 (g)
        add_bar_trace(growth_summary_df, 'avg_fresh_weight', '생중량', 1, 1)

        # 2. 평균 잎 수 (장)
        add_bar_trace(growth_summary_df, 'avg_leaf_count', '잎 수', 1, 2)

        # 3. 평균 지상부 길이 (mm)
        add_bar_trace(growth_summary_df, 'avg_shoot_length', '지상부 길이', 2, 1)

        # 4. 개체수
        add_bar_trace(growth_summary_df, 'count', '개체수', 2, 2)

        # 레이아웃 설정
        fig_growth_comp.update_layout(height=750, showlegend=False, 
                                      font=dict(family=PLOTLY_FONT),
                                      title_text="**EC별 생육 지표 비교**")
        fig_growth_comp.update_xaxes(title_text="EC 조건")
        
        st.plotly_chart(fig_growth_comp, use_container_width=True)

        st.markdown("---")

        # 3. 학교별 생중량 분포 (바이올린 플롯)
        st.subheader(f"📊 {selected_school} 생중량 분포 (바이올린 플롯)")
        
        # '전체' 선택 시 모든 학교, 개별 학교 선택 시 해당 학교만 표시
        if filtered_growth_df.empty:
            st.warning("선택한 학교의 생육 데이터가 없습니다.")
        else:
            if selected_school == "전체":
                fig_violin = px.violin(filtered_growth_df, y="fresh_weight", x="school", 
                                       box=True, points="all",
                                       color="school", 
                                       color_discrete_map={s: EC_MAPPING[s]['color'] for s in EC_MAPPING},
                                       title="학교별(EC별) 생중량 분포 비교",
                                       labels={"fresh_weight": "생중량 (g)", "school": "학교명 / EC 목표"})
            else:
                fig_violin = px.violin(filtered_growth_df, y="fresh_weight", 
                                       box=True, points="all",
                                       color_discrete_sequence=[EC_MAPPING[selected_school]['color']],
                                       title=f"{selected_school} 생중량 분포",
                                       labels={"fresh_weight": "생중량 (g)"})
                
            fig_violin.update_layout(font=dict(family=PLOTLY_FONT))
            st.plotly_chart(fig_violin, use_container_width=True)

        st.markdown("---")

        # 4. 상관관계 분석 (산점도 2개)
        st.subheader(f"📈 {selected_school} 생육 지표 간 상관관계")
        
        if not filtered_growth_df.empty:
            col_scatter_1, col_scatter_2 = st.columns(2)

            # 1. 잎 수 vs 생중량
            with col_scatter_1:
                fig_corr_leaf = px.scatter(filtered_growth_df, x='leaf_count', y='fresh_weight',
                                           color='school' if selected_school == "전체" else None,
                                           color_discrete_map={s: EC_MAPPING[s]['color'] for s in EC_MAPPING},
                                           trendline="ols",
                                           title="잎 수 vs 생중량",
                                           labels={'leaf_count': '잎 수 (장)', 'fresh_weight': '생중량 (g)'})
                fig_corr_leaf.update_layout(font=dict(family=PLOTLY_FONT))
                st.plotly_chart(fig_corr_leaf, use_container_width=True)

            # 2. 지상부 길이 vs 생중량
            with col_scatter_2:
                fig_corr_shoot = px.scatter(filtered_growth_df, x='shoot_length', y='fresh_weight',
                                           color='school' if selected_school == "전체" else None,
                                           color_discrete_map={s: EC_MAPPING[s]['color'] for s in EC_MAPPING},
                                           trendline="ols",
                                           title="지상부 길이 vs 생중량",
                                           labels={'shoot_length': '지상부 길이 (mm)', 'fresh_weight': '생중량 (g)'})
                fig_corr_shoot.update_layout(font=dict(family=PLOTLY_FONT))
                st.plotly_chart(fig_corr_shoot, use_container_width=True)
        else:
            st.info("선택된 학교의 생육 데이터가 부족하여 상관관계를 분석할 수 없습니다.")


        # 생육 데이터 원본
        with st.expander("원본 생육 결과 데이터 테이블 및 다운로드"):
            st.dataframe(filtered_growth_df.drop(columns=['ec_goal'], errors='ignore'), use_container_width=True)
            
            # XLSX 다운로드 (io.BytesIO 사용 - TypeError 방지)
            if not filtered_growth_df.empty:
                buffer = io.BytesIO()
                # 'openpyxl' 엔진을 사용하여 엑셀 파일 작성
                filtered_growth_df.to_excel(buffer, index=False, engine="openpyxl")
                buffer.seek(0) # 커서를 맨 앞으로 이동

                st.download_button(
                    label="다운로드: 생육 데이터 (XLSX)",
                    data=buffer,
                    file_name=f"{selected_school}_생육결과데이터_raw.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
