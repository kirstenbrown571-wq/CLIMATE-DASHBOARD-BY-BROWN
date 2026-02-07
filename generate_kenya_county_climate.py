import pandas as pd
import numpy as np

# -------------------------------
# CONSTANTS
# -------------------------------
COUNTIES = [
    "Baringo","Bomet","Bungoma","Busia","Elgeyo Marakwet","Embu","Garissa",
    "Homa Bay","Isiolo","Kajiado","Kakamega","Kericho","Kiambu","Kilifi",
    "Kirinyaga","Kisii","Kisumu","Kitui","Kwale","Laikipia","Lamu","Machakos",
    "Makueni","Mandera","Marsabit","Meru","Migori","Mombasa","Murang'a",
    "Nairobi","Nakuru","Nandi","Narok","Nyamira","Nyandarua","Nyeri",
    "Samburu","Siaya","Taita Taveta","Tana River","Tharaka Nithi",
    "Trans Nzoia","Turkana","Uasin Gishu","Vihiga","Wajir","West Pokot"
]

YEARS = range(1901, 2025)

CO2_1901 = 296
CO2_2024 = 421

rows = []

# -------------------------------
# GENERATE DATA
# -------------------------------
for county in COUNTIES:
    base_temp = np.random.uniform(17, 27)
    base_rain = np.random.uniform(450, 1400)
    base_humidity = np.random.uniform(55, 75)

    for year in YEARS:
        warming_trend = 0.014 * (year - 1901)
        temperature = base_temp + warming_trend + np.random.normal(0, 0.4)
        precipitation = base_rain + np.random.normal(0, 55)
        humidity = np.clip(base_humidity + np.random.normal(0, 6), 30, 90)
        wind = np.clip(np.random.normal(3.0, 0.7), 0.5, 7)
        solar_radiation = np.clip(5.8 + np.random.normal(0, 0.35), 4.8, 7.9)
        co2 = CO2_1901 + (year - 1901) * (CO2_2024 - CO2_1901) / (2024 - 1901)
        sunspots = int(60 + 45 * np.sin(2 * np.pi * (year % 11) / 11))

        rows.append([
            county,
            year,
            round(temperature, 2),
            round(precipitation, 1),
            round(humidity, 1),
            round(wind, 2),
            round(solar_radiation, 2),
            round(co2, 1),
            sunspots
        ])

# -------------------------------
# SAVE CSV
# -------------------------------
df = pd.DataFrame(rows, columns=[
    "Location",
    "Year",
    "Temperature",
    "Precipitation",
    "Humidity",
    "Wind",
    "SolarRadiation",
    "CO2",
    "Sunspots"
])

df.to_csv("Kenya_county_climate.csv", index=False, encoding="utf-8")
print("✅ Kenya_county_climate.csv generated successfully (1901–2024, all counties)")