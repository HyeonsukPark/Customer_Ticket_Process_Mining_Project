import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pm4py.statistics.variants.log import get as variants_module
from pm4py.objects.conversion.log import converter as log_converter

load_dotenv('.env')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title = "Process Mining + LLM Insights", layout="wide")
st.title("Process Mining + LLM Insights Dashboard")
st.markdown("Analyze process performance, satisfaction, and AI insights.")

# ========= Load Data ============== #

uploaded_file = st.file_uploader("Upload your event log CSV file", type=["csv"])

if uploaded_file:
    log_df = pd.read_csv(uploaded_file)
else:
    st.info("Upload your process mining dataset")
    st.stop()

# ======= Data preprocess =========== #

numeric_cols = ['case_duration', 'trace:customer_satisfaction']
cat_cols = ['concept:name', 'trace:issue_type', 'event:resolver']
log_df["time:timestamp"] = pd.to_datetime(log_df["time:timestamp"])

# ======= Basic cleaning ============= #

log_df = log_df.dropna(subset=numeric_cols)
log_df = log_df.sort_values(by="case_duration")

# ======= correlation ======= #

corr_value = log_df['case_duration'].corr(log_df['trace:customer_satisfaction'])
# st.metric("correlation (Duration vs Satisfaction)", f"{corr_value:.2f}")

summary = (
    log_df.groupby(cat_cols)[numeric_cols]
    .mean()
    .reset_index()
)

# ======== visualization ====== #

tab1, tab2, tab3, tab4 = st.tabs(["correlation Plot", "Satisfaction by Activity", "Issue Type Analysis", 'Resolver Analysis'])

with tab1:
    st.subheader("Case Duration vs Satisfaction Score")
    fig1 = px.scatter(
        log_df, x="case_duration", y="trace:customer_satisfaction", color="trace:issue_type",
        hover_data = ['concept:name', 'event:resolver'], title = "Case Duration vs Satisfaction"
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Average Satisfaction by Activity")
    fig2 = px.bar(log_df.groupby('concept:name')['trace:customer_satisfaction'].mean().reset_index(),
           x = 'concept:name', y = 'trace:customer_satisfaction', color='concept:name',
           title='Average Satisfaction by Activity'
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Average Satisfaction by Issue Type")
    fig3 = px.bar(log_df.groupby('trace:issue_type')['trace:customer_satisfaction'].mean().reset_index(),
                  x='trace:issue_type', y='trace:customer_satisfaction', color='trace:issue_type',
                  title="Satisfaction Distribution per Issue Type")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Average Satisfaction by Resolver")
    fig2 = px.bar(log_df.groupby('event:resolver')['trace:customer_satisfaction'].mean().reset_index(),
           x = 'event:resolver', y = 'trace:customer_satisfaction', color='event:resolver',
           title='Average Satisfaction by Resolver'
    )
    st.plotly_chart(fig2, use_container_width=True)


# ======== Variants ======== #
variants_count = variants_module.get_variants(log_df)
variant_stats = []

for variant_name, traces in variants_count.items():
    case_ids = [trace.attributes['concept:name'] for trace in traces]

    # Filter out cases with only one event before calculating duration
    variant_cases = log_df[log_df['case:concept:name'].isin(case_ids)].groupby('case:concept:name').filter(lambda x: len(x) > 1)

    avg_satisfaction = log_df[log_df['case:concept:name'].isin(case_ids)]['trace:customer_satisfaction'].mean()

    avg_duration = variant_cases.groupby('case:concept:name')['time:timestamp'].apply(
        lambda x: (x.max() - x.min()).total_seconds() / 3600
        ).mean()

    variant_stats.append({
        'Variant': variant_name,
        'Frequency (#Cases)': len(traces),
        'Avg Satisfaction': round(avg_satisfaction, 2) if not pd.isna(avg_satisfaction) else None,
        'Avg Duration (hrs)': round(avg_duration, 2) if not pd.isna(avg_duration) else None
    })

variants_df = pd.DataFrame(variant_stats).sort_values(by='Frequency (#Cases)', ascending=False)

st.subheader("Variant Summary")
st.dataframe(variants_df)

# ============ variants summary ==================#
st.subheader("Automated Insights Summary")

# Identify key variants
top_variant = variants_df.iloc[0]
low_sat_variant = variants_df.loc[variants_df["Avg Satisfaction"].idxmin()]
high_sat_variant = variants_df.loc[variants_df["Avg Satisfaction"].idxmax()]

# Create summary text
insight_text = f"""
### Key Variant Insights

- **Most Frequent Variant:**  
  **{top_variant['Variant']}** occurs in **{top_variant['Frequency (#Cases)']} cases** 
  with an average satisfaction of **{top_variant['Avg Satisfaction']}**  
  and average duration of **{top_variant['Avg Duration (hrs)']} hrs**.

- **Highest Satisfaction Variant:**  
  **{high_sat_variant['Variant']}** shows the **highest customer satisfaction**  
  at **{high_sat_variant['Avg Satisfaction']}**, with an average duration of **{high_sat_variant['Avg Duration (hrs)']} hrs**  
  across **{high_sat_variant['Frequency (#Cases)']} cases**.

- **Lowest Satisfaction Variant:**  
  **{low_sat_variant['Variant']}** has the **lowest satisfaction** (score **{low_sat_variant['Avg Satisfaction']}**)  
  and takes an average of **{low_sat_variant['Avg Duration (hrs)']} hrs** to complete  
  across **{low_sat_variant['Frequency (#Cases)']} cases**.
"""

st.markdown(insight_text)

# ====== LLM Insights ========= #

st.markdown("---")
st.header("LLM-Generated Insights")

with st.spinner("Generating Insights..."):
    summary_table = summary.head(15).to_markdown(index=False)
    prompt = f"""
    You are an expert data analyst specialized in process mining and customer experience.
    Analyze the following summary data from a process mining dataset.
    
    {summary_table}
    
    The correlation between case duration and satisfaction score is {corr_value:.2f}.
    Summarize the main findings and explain which combinations or patterns lead to lower satisfaction.
    Provide concise, actionable insights in bullet points.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        llm_summary = response.choices[0].message.content
        st.success("Insights generated successfully!")
    except Exception as e:
        llm_summary = f"Error generating insights: {e}"

    st.markdown(llm_summary)
