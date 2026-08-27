import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Car Market Trends Analysis",
    page_icon="🚗",
    layout="wide"
)

# Global Aesthetics
sns.set_theme(style="whitegrid")

# App Header
st.title("🚗 Car Market Trends Analysis Dashboard")
st.write("An interactive dashboard analyzing vehicle depreciation, fuel pricing trends, and market distributions.")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("Car data.csv")

try:
    df = load_data()
    
    # Overview Metrics & Raw Data Toggle
    st.subheader("Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cars", len(df))
    col2.metric("Avg Selling Price", f"₹{df['Selling_Price'].mean():.2f} L")
    col3.metric("Avg Present Price", f"₹{df['Present_Price'].mean():.2f} L")
    col4.metric("Avg Kms Driven", f"{df['Kms_Driven'].mean():,.0f} km")

    if st.checkbox("Show Raw Dataset"):
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.header("Visualizations")

    # Layout Row 1
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("1. Average Selling Price Trend by Year")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        yearly_price = df.groupby("Year")["Selling_Price"].mean().reset_index()
        sns.lineplot(data=yearly_price, x="Year", y="Selling_Price", marker="o", color="navy", ax=ax1)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Selling Price (Lakhs)")
        st.pyplot(fig1)

    with row1_col2:
        st.subheader("2. Average Selling Price by Fuel Type")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=df,
            x="Fuel_Type",
            y="Selling_Price",
            estimator="mean",
            errorbar=None,
            hue="Fuel_Type",
            legend=False,
            palette="Blues_d",
            ax=ax2
        )
        ax2.set_ylabel("Selling Price (Lakhs)")
        st.pyplot(fig2)

    # Layout Row 2
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("3. Distribution of Vehicle Fuel Types")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        df["Fuel_Type"].value_counts().plot.pie(
            autopct="%1.1f%%", 
            colors=["#66b3ff", "#ff9999", "#99ff99"], 
            ax=ax3
        )
        ax3.set_ylabel("")
        st.pyplot(fig3)

    with row2_col2:
        st.subheader("4. Selling Price Distribution")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.histplot(df["Selling_Price"], kde=True, bins=20, color="teal", ax=ax4)
        ax4.set_xlabel("Selling Price (Lakhs)")
        st.pyplot(fig4)

    # Layout Row 3
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        st.subheader("5. Present Price vs. Selling Price")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df, 
            x="Present_Price", 
            y="Selling_Price", 
            hue="Fuel_Type", 
            style="Transmission", 
            s=70, 
            ax=ax5
        )
        ax5.set_xlabel("Present Price (Lakhs)")
        ax5.set_ylabel("Selling Price (Lakhs)")
        st.pyplot(fig5)

    with row3_col2:
        st.subheader("6. Vehicle Count by Seller & Transmission")
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x="Seller_Type", hue="Transmission", palette="Set2", ax=ax6)
        ax6.set_xlabel("Seller Type")
        ax6.set_ylabel("Vehicle Count")
        st.pyplot(fig6)

    # Layout Row 4
    row4_col1, row4_col2 = st.columns(2)

    with row4_col1:
        st.subheader("7. Price Range by Transmission Type")
        fig7, ax7 = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x="Transmission",
            y="Selling_Price",
            hue="Transmission",
            legend=False,
            palette="Set3",
            ax=ax7
        )
        ax7.set_ylabel("Selling Price (Lakhs)")
        st.pyplot(fig7)

    with row4_col2:
        st.subheader("8. Numerical Feature Correlation Matrix")
        fig8, ax8 = plt.subplots(figsize=(8, 5))
        numeric_df = df.select_dtypes(include=["number"])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax8)
        st.pyplot(fig8)

except FileNotFoundError:
    st.error("Error: 'Car data.csv' not found. Make sure the file is in the same directory as this script.")
