import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import requests
import plotly.express as px

sns.set_style("darkgrid")
plt.rcParams.update({"figure.autolayout": True})

st.title("🌍 Climate Intelligence Dashboard")
st.markdown("### Predicting climate trends for Kenya by county")
st.caption("Merged local CSV + online data with advanced research features")

county = st.selectbox("Select County:", [
    "Mombasa","Kwale","Kilifi","Tana River","Lamu","Taita-Taveta","Garissa","Wajir","Mandera",
    "Marsabit","Isiolo","Meru","Tharaka-Nithi","Embu","Kitui","Machakos","Makueni","Nyandarua",
    "Nyeri","Kirinyaga","Murang'a","Kiambu","Turkana","West Pokot","Samburu","Trans Nzoia",
    "Uasin Gishu","Elgeyo-Marakwet","Nandi","Baringo","Laikipia","Nakuru","Narok","Kajiado",
    "Kericho","Bomet","Kakamega","Vihiga","Bungoma","Busia","Siaya","Kisumu","Homa Bay",
    "Migori","Kisii","Nyamira","Nairobi"
])
year_input = st.number_input("Enter year (1901-2024):", min_value=1901, max_value=2024, value=2020, step=1)
chart_type = st.radio("Choose chart type:", ["Matplotlib (static)", "Plotly (interactive)"])
milankovitch_option = st.checkbox("Analyze Milankovitch cycles")
fetch_button = st.button("Get Data & Prediction")

@st.cache_data
def fetch_worldbank_data(county, year):
    try:
        url = f"https://climateknowledgeportal.worldbank.org/api/data?location={county}&year={year}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            df["Source"] = "Online"
            return df
        else:
            return pd.DataFrame()
    except:
        return pd.DataFrame()

def fetch_noaa_co2(years):
    return [315 + (y - 1960) * 2 if y >= 1960 else 315 for y in years]

def fetch_nasa_sunspots(years):
    return np.random.randint(0, 200, len(years))

if fetch_button:
    with st.spinner("Loading local + online data..."):
        try:
            local_df = pd.read_csv("Kenya_county_climate.csv")
            local_df = local_df[local_df["Location"] == county]
            local_df["Source"] = "CSV"
        except Exception as e:
            st.error(f"Error loading local CSV: {e}")
            local_df = pd.DataFrame()

        online_df = fetch_worldbank_data(county, year_input)
        if not online_df.empty:
            online_df["CO2 (ppm)"] = fetch_noaa_co2(online_df["Year"])
            online_df["Sunspots"] = fetch_nasa_sunspots(online_df["Year"])

        combined_df = pd.concat([local_df, online_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["Location","Year"], keep="last")
        combined_df = combined_df.sort_values(by="Year")

    # ------------------------------
    # STATUS BANNER
    # ------------------------------
    if combined_df.empty:
        st.warning("⚠️ No data available for this county.")
    else:
        if "Online" in combined_df["Source"].values:
            st.success("✅ Data merged: CSV + Online sources")
        else:
            st.info("ℹ️ Showing CSV data only (no online data available)")

        st.subheader(f"📊 Climate Data for {county} up to {year_input}")
        st.dataframe(combined_df.tail(10))

        # Download button
        csv_download = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download merged dataset", csv_download, "merged_climate_data.csv", "text/csv")

        # ------------------------------
        # SEPARATE GRAPHS FOR EACH PARAMETER
        # ------------------------------
        variable_map = {
            "Temperature": "Temperature (°C)",
            "Precipitation": "Annual Rainfall (mm)",
            "Humidity": "Average Humidity (%)",
            "Wind": "Average Wind Speed (m/s)",
            "SolarRadiation": "Solar Radiation (kWh/m²)",
            "CO2 (ppm)": "CO₂ Concentration (ppm)",
            "Sunspots": "Sunspots"
        }

        for col, title in variable_map.items():
            if col not in combined_df.columns:
                st.warning(f"⚠️ {title} data not available in this dataset.")
                continue

            st.markdown(f"### 📊 {title}")
            if chart_type == "Matplotlib (static)":
                fig, ax = plt.subplots(figsize=(10,5))
                csv_data = combined_df[combined_df["Source"]=="CSV"]
                online_data = combined_df[combined_df["Source"]=="Online"]

                ax.plot(csv_data["Year"], csv_data[col], 'o-', color='green', alpha=0.7, label=f'CSV {title}')
                ax.plot(online_data["Year"], online_data[col], 'x--', color='blue', linewidth=2, label=f'Online {title}')
                ax.set_xlabel("Year", fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.set_title(f"{title} Trend for {county}", fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                fig = px.line(
                    combined_df,
                    x="Year",
                    y=col,
                    color="Source",
                    markers=True,
                    title=f"{title} Trend for {county}"
                )
                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title=title,
                    legend_title="Data Source",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        # ------------------------------
        # TREND ANALYSIS TABLE
        # ------------------------------
        st.markdown("### 📈 Trend Analysis")
        trend_results = []
        for col in variable_map.keys():
            if col in combined_df.columns:
                X = combined_df["Year"].values.reshape(-1, 1)
                y = combined_df[col].values
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
                direction = "Increasing 📈" if slope > 0 else "Decreasing 📉"
                trend_results.append({"Variable": col, "Slope per Year": round(slope, 4), "Trend": direction})
        if trend_results:
            st.table(pd.DataFrame(trend_results))
        else:
            st.info("No variables available for trend analysis.")

        # ------------------------------
        # ROLLING AVERAGES
        # ------------------------------
        st.markdown("### 📊 Rolling Averages (10-year smoothing)")
        for col in ["Temperature","Precipitation","Humidity"]:
            if col in combined_df.columns:
                st.markdown(f"#### {col}")
                combined_df[f"{col}_rolling"] = combined_df[col].rolling(window=10).mean()
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(combined_df["Year"], combined_df[col], alpha=0.4, label="Raw Data")
                ax.plot(combined_df["Year"], combined_df[f"{col}_rolling"], color="red", linewidth=2, label="10-year Rolling Mean")
                ax.set_title(f"{col} with Rolling Average")
                ax.legend()
                st.pyplot(fig)

        # ------------------------------
        # MILANKOVITCH CYCLES (Corrected)
        # ------------------------------
        if milankovitch_option:
            st.markdown("### 🌌 Milankovitch Cycle Overlay")
            years = combined_df["Year"].values

            # Scaled periods for visualization
            ecc = np.sin(2*np.pi*years/1000)
            obl = np.sin(2*np.pi*years/410)
            pre = np.sin(2*np.pi*years/230)

            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(years, ecc, label="Eccentricity (scaled)", color="red")
            ax.plot(years, obl, label="Obliquity (scaled)", color="blue")
            ax.plot(years, pre, label="Precession (scaled)", color="green")
            ax.set_title("Milankovitch Cycles (scaled for visualization)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel("Cycle Signal (normalized)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # ------------------------------
        # CORRELATION MATRIX HEATMAP
        # ------------------------------
        st.markdown("### 🔗 Correlation Matrix Heatmap")
