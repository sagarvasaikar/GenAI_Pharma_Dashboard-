# genai_dashboard.py

# --- Imports ---
import streamlit as st 
import pandas as pd  
import numpy as np  
from sklearn.metrics.pairwise import cosine_similarity 
import seaborn as sns  
import matplotlib.pyplot as plt 
from openai import OpenAI  
import key 
import plotly.express as px  
import plotly.graph_objects as go

# --- Set up OpenAI client using API key 
client = OpenAI(api_key=key.OPENAI_API_KEY)

# --- Load and prepare data from Excel file ---
@st.cache_data  # Cache the function to avoid reloading on every run
def load_data(path):
    df = pd.read_excel(path)  # Load Excel data
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")  # Clean column names
    df = df[df["comments"].notna() & df["comments"].str.strip().ne("")].copy()  # Drop rows with empty comments
    df["site"] = df["organization_name"]  # Create a new column "site"
    return df

# Load dataset
df = load_data("C:/Users/yashu/genai_poc/dummy_nc_capa_dataset1.xlsx")

# --- Sidebar filters for dashboard ---
selected_site = st.sidebar.selectbox("Select Site", ["All"] + sorted(df["site"].unique().tolist()))
selected_sector = st.sidebar.selectbox("Select Sector", ["All"] + sorted(df["sector"].dropna().unique().tolist()))
selected_process_area = st.sidebar.selectbox("Select Process Area", ["All"] + sorted(df["process_area"].dropna().unique().tolist()))
selected_kpi = st.sidebar.selectbox("Select KPI", ["All"] + sorted(df["kpi_name"].dropna().unique().tolist()))
selected_period = st.sidebar.selectbox("Select Time Period", ["All"] + sorted(df["reporting_period"].astype(str).unique().tolist()))

# Apply filters
filtered_df = df.copy()
if selected_site != "All":
    filtered_df = filtered_df[filtered_df["site"] == selected_site]
if selected_sector != "All":
    filtered_df = filtered_df[filtered_df["sector"] == selected_sector]
if selected_process_area != "All":
    filtered_df = filtered_df[filtered_df["process_area"] == selected_process_area]
if selected_kpi != "All":
    filtered_df = filtered_df[filtered_df["kpi_name"] == selected_kpi]
if selected_period != "All":
    filtered_df = filtered_df[filtered_df["reporting_period"].astype(str) == selected_period]

# --- Define predefined root cause categories ---
CATEGORY_LIST = [
    "Documentation Error",
    "Human Error",
    "Equipment Failure",
    "Training Deficiency",
    "SOP Deviation",
    "Supplier Issue"
]

# Classify comments using GPT model
@st.cache_data
def classify_comment(comment):
    prompt = f"""You are a pharmaceutical quality compliance assistant.
Classify the following comment into one of these categories:
{', '.join(CATEGORY_LIST)}.

Comment: \"{comment}\"
Category:"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()  # Return category
    except Exception as e:
        return "Error"  # Return error if API call fails

# Run classification only if not already classified
if "category" not in filtered_df.columns:
    filtered_df = filtered_df.sample(min(50, len(filtered_df)), random_state=42).copy()
    with st.spinner("Classifying comments..."):
        filtered_df["category"] = filtered_df["comments"].apply(classify_comment)

# --- Summarize individual comments ---
@st.cache_data
def summarize_single_comment(comment):
    prompt = f"""You are a pharmaceutical quality assistant.
Summarize the following quality-related comment in 1-2 sentences:

Comment: "{comment}"
Summary:"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Summary error"

with st.spinner("Generating summaries for each comment..."):
    filtered_df["comment_summary"] = filtered_df["comments"].apply(summarize_single_comment)

# --- GPT Summary per site and process ---
@st.cache_data
def summarize_comments(site, process_area, comments):
    text = "\n".join(f"- {c}" for c in comments[:10])  # Use only top 10 comments
    prompt = f"""Summarize key quality issue themes from these comments at site '{site}' in process area '{process_area}':
{text}
Summary:"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Summary error"

# Generate summaries by group
summary_rows = []
for (site, area), group in filtered_df.groupby(["site", "process_area"]):
    summary = summarize_comments(site, area, group["comments"].tolist())
    summary_rows.append({
        "site": site, "process_area": area, "summary": summary
    })
summary_df = pd.DataFrame(summary_rows)

# --- Get embedding for comments ---
def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding  # Return embedding vector
    except Exception as e:
        return [0.0] * 1536  # Return default embedding if error

# Generate embeddings if not already present
if "embedding" not in filtered_df.columns:
    with st.spinner("Generating embeddings..."):
        filtered_df["embedding"] = filtered_df["comments"].apply(get_embedding)

# --- Calculate pairwise similarity between embeddings ---
similarities = []
for i in range(len(filtered_df)):
    for j in range(i + 1, len(filtered_df)):
        sim = cosine_similarity([filtered_df.iloc[i]["embedding"]], [filtered_df.iloc[j]["embedding"]])[0][0]
        if sim > 0.85:  # Only store highly similar comment pairs
            similarities.append({
                "site_1": filtered_df.iloc[i]["site"],
                "site_2": filtered_df.iloc[j]["site"],
                "comment_1": filtered_df.iloc[i]["comments"],
                "comment_2": filtered_df.iloc[j]["comments"],
                "similarity": sim
            })

similar_df = pd.DataFrame(similarities)  # Store similarities in dataframe

# ============================
# -------- UI Layout ---------
# ============================

st.title("NC/CAPA Executive Dashboard")

# --- Root Cause Category Distribution (Bar Chart) ---
st.subheader("\U0001F4CA Root Cause Category Distribution")
fig, ax = plt.subplots()
sns.countplot(y="category", data=filtered_df, order=filtered_df["category"].value_counts().index, ax=ax)
st.pyplot(fig)

# --- Treemap by Site and Process Area ---
st.subheader("\U0001F4DC Issue Volume by Site and Process Area")
tree_df = filtered_df.groupby(["site", "process_area"]).size().reset_index(name="count")
tree_df["label"] = tree_df["site"] + " - " + tree_df["process_area"]
fig_tree = px.treemap(tree_df, path=["site", "process_area"], values="count", color="count", color_continuous_scale="Blues")
st.plotly_chart(fig_tree, use_container_width=True)


# --- Category Trend Over Time 
st.subheader("\U0001F5D3 Category Trends Over Time")
temp_df = filtered_df.copy()
temp_df["year"] = pd.to_datetime(temp_df["reporting_period"]).dt.year

# Dropdown for time granularity
time_level = st.selectbox("Select Time Granularity", ["Year", "Quarter", "Month"])

if time_level == "Year":
    temp_df["time_period"] = temp_df["year"].astype(str)
elif time_level == "Quarter":
    temp_df["time_period"] = pd.to_datetime(temp_df["reporting_period"]).dt.to_period("Q").astype(str)
else:
    temp_df["time_period"] = pd.to_datetime(temp_df["reporting_period"]).dt.to_period("M").astype(str)

count_df = temp_df.groupby(["time_period", "category"]).size().reset_index(name="count")
fig_bar = px.bar(count_df, x="time_period", y="count", color="category", barmode="group")
st.plotly_chart(fig_bar, use_container_width=True)

# --- GPT Summary Cards ---
st.subheader("\U0001F4DD Summarized Insights by Site & Process")
for i, row in summary_df.iterrows():
    with st.expander(f"Summary for {row['site']} - {row['process_area']}"):
        st.markdown(row["summary"])

# --- Display summarized comments ---
#st.subheader("🧠 AI-Generated Summaries for Each Comment")
#st.dataframe(filtered_df[["site", "process_area", "category", "comments", "comment_summary"]], use_container_width=True)


# --- Similarity Heatmap between sites ---
st.subheader("\U0001F30D Global Similarity Heatmap (Issue Overlap)")
if not similar_df.empty:
    pivot = pd.pivot_table(similar_df, index="site_1", columns="site_2", values="similarity", aggfunc="mean")
    fig2, ax2 = plt.subplots()
    sns.heatmap(pivot, cmap="coolwarm", annot=True, fmt=".2f", ax=ax2)
    st.pyplot(fig2)
else:
    st.info("No high-similarity comment pairs found.")

# --- Comment Table with Classification ---
#st.subheader("\U0001F50E Drill-Down Comments with Classification")
#st.dataframe(filtered_df[["site", "process_area", "category", "comments"]].reset_index(drop=True), use_container_width=True)

# --- Export scorecard CSV ---
#st.subheader("\U0001F4C4 Download Scorecard")
#scorecard_export = filtered_df[["site", "process_area", "kpi_name", "comments", "category"]]
#st.download_button("Download Excel", scorecard_export.to_csv(index=False), file_name="scorecard.csv")


# --- Display DataFrames ---
st.subheader("📂 Raw Filtered Dataset")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

#st.subheader("📝 Summary DataFrame")
#st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)

st.subheader("🔍 Similar Comments DataFrame")
st.dataframe(similar_df.reset_index(drop=True), use_container_width=True)

# --- Summary Table with Expandable Rows (Option 4) ---
#st.subheader("📋 Summary Table by Site & Process Area")
#for i, row in summary_df.iterrows():
#    st.markdown(f"**{row['site']} - {row['process_area']}**")
#    st.text_area(
#        label="Summary:",
#        value=row['summary'],
#        height=100,
#        key=f"summary_{i}"
#    )

# --- Summary Accordions with Category Tags (Option 6) ---
#st.subheader("📂 Interactive Summary Explorer")
#for i, row in summary_df.iterrows():
#    with st.expander(f"📌 {row['site']} → {row['process_area']}"):
#        st.markdown(f"**Summary:** {row['summary']}")
#        # Optional: Display additional tags or stats below if needed

# genai_dashboard.py

# [All previous code remains unchanged above this line]

# --- Similarity Visuals ---
#st.subheader("\U0001F310 Site Similarity Visuals")

# # Chord-style Sankey Diagram (Option 2)
# if not similar_df.empty:
#     sankey_df = similar_df.copy()
#     sankey_fig = go.Figure(data=[
#         go.Sankey(
#             node=dict(
#                 pad=15,
#                 thickness=20,
#                 line=dict(color="black", width=0.5),
#                 label=list(set(sankey_df["site_1"]).union(set(sankey_df["site_2"]))),
#                 color="blue"
#             ),
#             link=dict(
#                 source=[list(set(sankey_df["site_1"]).union(set(sankey_df["site_2"]))).index(s) for s in sankey_df["site_1"]],
#                 target=[list(set(sankey_df["site_1"]).union(set(sankey_df["site_2"]))).index(t) for t in sankey_df["site_2"]],
#                 value=sankey_df["similarity"]
#             )
#         )
#     ])
#     sankey_fig.update_layout(title_text="Sankey Diagram of Site Similarities", font_size=10)
#     st.plotly_chart(sankey_fig, use_container_width=True)

# # Network Graph (Option 3)
# import networkx as nx
# import matplotlib.pyplot as plt

# if not similar_df.empty:
#     G = nx.Graph()
#     for _, row in similar_df.iterrows():
#         G.add_edge(row['site_1'], row['site_2'], weight=row['similarity'])

#     plt.figure(figsize=(10, 6))
#     pos = nx.spring_layout(G, seed=42)
#     edges = G.edges(data=True)
#     weights = [d['weight'] for (_, _, d) in edges]
#     nx.draw(G, pos, with_labels=True, width=weights, edge_color=weights, edge_cmap=plt.cm.Blues, node_color='lightblue')
#     st.pyplot(plt.gcf())

# # Stacked Bar Chart of Similarity Counts per Site (Option 4)
# count_matrix = similar_df.groupby(["site_1", "site_2"]).size().reset_index(name="count")
# fig_stack = px.bar(
#     count_matrix,
#     x="site_1",
#     y="count",
#     color="site_2",
#     title="Stacked Bar of Similarity Counts per Site",
#     labels={"site_1": "Source Site", "site_2": "Similar To", "count": "# of Similar Comments"},
#     barmode="stack"
# )
# st.plotly_chart(fig_stack, use_container_width=True)

# Bubble Chart for Site Pairs (Option 5)
similar_df['pair'] = similar_df['site_1'] + ' ↔ ' + similar_df['site_2']
fig_bubble = px.scatter(
    similar_df,
    x="site_1",
    y="site_2",
    size="similarity",
    color="similarity",
    hover_name="pair",
    title="Bubble Chart of Similar Site Comment Pairs",
    size_max=30,
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_bubble, use_container_width=True)

# # Interactive Table with Highlight (Option 6)
# st.subheader("\U0001F5C3 Similar Comment Table Viewer")
# st.dataframe(similar_df[["site_1", "site_2", "similarity", "comment_1", "comment_2"]], use_container_width=True)

# --- Display All Columns Including Category and Summary ---
st.subheader("📘 Full Dataset with Classification and Summaries")
if "category" in filtered_df.columns and "comment_summary" in filtered_df.columns:
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)
else:
    st.warning("Category or summary columns not found. Please ensure classification and summarization steps are complete.")

# Command to run: streamlit run genai_poc.py
