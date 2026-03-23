"""
Economics & Poverty Dataset Generator
Generates synthetic but realistic socioeconomic data for 50 countries (2015-2023)
"""
import pandas as pd
import numpy as np
import sqlite3

np.random.seed(42)

COUNTRIES = {
    "United States":  {"region": "North America",  "dev": "high"},
    "Canada":         {"region": "North America",  "dev": "high"},
    "Mexico":         {"region": "North America",  "dev": "middle"},
    "Brazil":         {"region": "South America",  "dev": "middle"},
    "Argentina":      {"region": "South America",  "dev": "middle"},
    "Colombia":       {"region": "South America",  "dev": "middle"},
    "Peru":           {"region": "South America",  "dev": "lower-middle"},
    "Chile":          {"region": "South America",  "dev": "middle"},
    "Venezuela":      {"region": "South America",  "dev": "lower-middle"},
    "Bolivia":        {"region": "South America",  "dev": "lower-middle"},
    "United Kingdom": {"region": "Europe",         "dev": "high"},
    "Germany":        {"region": "Europe",         "dev": "high"},
    "France":         {"region": "Europe",         "dev": "high"},
    "Italy":          {"region": "Europe",         "dev": "high"},
    "Spain":          {"region": "Europe",         "dev": "high"},
    "Poland":         {"region": "Europe",         "dev": "high"},
    "Ukraine":        {"region": "Europe",         "dev": "lower-middle"},
    "Romania":        {"region": "Europe",         "dev": "middle"},
    "Greece":         {"region": "Europe",         "dev": "high"},
    "Portugal":       {"region": "Europe",         "dev": "high"},
    "China":          {"region": "Asia",           "dev": "middle"},
    "India":          {"region": "Asia",           "dev": "lower-middle"},
    "Japan":          {"region": "Asia",           "dev": "high"},
    "South Korea":    {"region": "Asia",           "dev": "high"},
    "Indonesia":      {"region": "Asia",           "dev": "lower-middle"},
    "Pakistan":       {"region": "Asia",           "dev": "lower-middle"},
    "Bangladesh":     {"region": "Asia",           "dev": "lower-middle"},
    "Vietnam":        {"region": "Asia",           "dev": "lower-middle"},
    "Thailand":       {"region": "Asia",           "dev": "middle"},
    "Philippines":    {"region": "Asia",           "dev": "lower-middle"},
    "Nigeria":        {"region": "Africa",         "dev": "lower-middle"},
    "Ethiopia":       {"region": "Africa",         "dev": "low"},
    "Egypt":          {"region": "Africa",         "dev": "lower-middle"},
    "DR Congo":       {"region": "Africa",         "dev": "low"},
    "Tanzania":       {"region": "Africa",         "dev": "low"},
    "Kenya":          {"region": "Africa",         "dev": "lower-middle"},
    "South Africa":   {"region": "Africa",         "dev": "middle"},
    "Ghana":          {"region": "Africa",         "dev": "lower-middle"},
    "Mozambique":     {"region": "Africa",         "dev": "low"},
    "Uganda":         {"region": "Africa",         "dev": "low"},
    "Saudi Arabia":   {"region": "Middle East",    "dev": "high"},
    "Turkey":         {"region": "Middle East",    "dev": "middle"},
    "Iran":           {"region": "Middle East",    "dev": "middle"},
    "Iraq":           {"region": "Middle East",    "dev": "middle"},
    "Israel":         {"region": "Middle East",    "dev": "high"},
    "Australia":      {"region": "Oceania",        "dev": "high"},
    "New Zealand":    {"region": "Oceania",        "dev": "high"},
    "Papua New Guinea":{"region":"Oceania",        "dev": "lower-middle"},
    "Russia":         {"region": "Europe",         "dev": "middle"},
    "Kazakhstan":     {"region": "Asia",           "dev": "middle"},
}

GDP_BASE = {
    "high": 45000, "middle": 12000, "lower-middle": 4000, "low": 1200
}
POVERTY_BASE = {
    "high": 0.12, "middle": 0.28, "lower-middle": 0.45, "low": 0.65
}
GINI_BASE = {
    "high": 0.32, "middle": 0.42, "lower-middle": 0.38, "low": 0.43
}
UNEMPLOYMENT_BASE = {
    "high": 0.055, "middle": 0.09, "lower-middle": 0.10, "low": 0.13
}
LITERACY_BASE = {
    "high": 0.98, "middle": 0.90, "lower-middle": 0.80, "low": 0.60
}
LIFE_EXP_BASE = {
    "high": 80, "middle": 73, "lower-middle": 68, "low": 62
}
HEALTH_EXP_BASE = {
    "high": 9.5, "middle": 6.0, "lower-middle": 4.2, "low": 2.8
}
EDU_EXP_BASE = {
    "high": 5.5, "middle": 4.2, "lower-middle": 3.5, "low": 2.8
}

records = []
for country, info in COUNTRIES.items():
    dev = info["dev"]
    region = info["region"]
    gdp_base = GDP_BASE[dev]

    for year in range(2015, 2024):
        t = year - 2015
        growth_rate = np.random.normal(0.025, 0.02)
        if year == 2020: growth_rate = np.random.normal(-0.04, 0.03)
        if year == 2021: growth_rate = np.random.normal(0.05, 0.02)

        gdp = gdp_base * ((1 + growth_rate) ** t) * np.random.uniform(0.92, 1.08)
        gdp = max(gdp, 500)

        poverty = max(0.01, POVERTY_BASE[dev] - t * 0.008 + np.random.normal(0, 0.02))
        if year == 2020: poverty += np.random.uniform(0.01, 0.04)

        gini = GINI_BASE[dev] + np.random.normal(0, 0.015)
        gini = np.clip(gini, 0.22, 0.65)

        unemployment = max(0.01, UNEMPLOYMENT_BASE[dev] + np.random.normal(0, 0.01))
        if year == 2020: unemployment += np.random.uniform(0.01, 0.05)
        if year >= 2021: unemployment = max(0.01, unemployment - 0.01)

        literacy = min(0.999, LITERACY_BASE[dev] + t * 0.002 + np.random.normal(0, 0.01))
        life_exp = min(88, LIFE_EXP_BASE[dev] + t * 0.15 + np.random.normal(0, 0.5))
        health_exp = HEALTH_EXP_BASE[dev] + np.random.normal(0, 0.4)
        edu_exp = EDU_EXP_BASE[dev] + np.random.normal(0, 0.3)
        internet = min(0.99, (0.35 if dev == "low" else 0.55 if dev == "lower-middle" else 0.72 if dev == "middle" else 0.89) + t * 0.02 + np.random.normal(0, 0.03))
        population = np.random.randint(5_000_000, 1_400_000_000)
        inflation = max(0.1, np.random.normal(3.5 if dev in ("high","middle") else 7.0, 2.5))
        if year == 2022: inflation += np.random.uniform(1, 5)

        hdi = round(0.3 * (life_exp - 20) / (85 - 20) +
                    0.3 * literacy +
                    0.4 * np.log(gdp) / np.log(75000), 3)
        hdi = np.clip(hdi, 0.30, 0.98)

        records.append({
            "country":          country,
            "region":           region,
            "income_group":     dev,
            "year":             year,
            "gdp_per_capita":   round(gdp, 2),
            "gdp_growth_pct":   round(growth_rate * 100, 2),
            "poverty_rate":     round(poverty, 4),
            "gini_index":       round(gini, 4),
            "unemployment_rate":round(unemployment, 4),
            "literacy_rate":    round(literacy, 4),
            "life_expectancy":  round(life_exp, 2),
            "health_exp_pct_gdp":round(np.clip(health_exp, 1, 15), 2),
            "edu_exp_pct_gdp":  round(np.clip(edu_exp, 1, 12), 2),
            "internet_access":  round(internet, 4),
            "inflation_rate":   round(inflation, 2),
            "population":       population,
            "hdi":              hdi,
        })

df = pd.DataFrame(records)
df.to_csv("/home/claude/econ_analytics/data/econ_data.csv", index=False)

conn = sqlite3.connect("/home/claude/econ_analytics/data/econ.db")
df.to_sql("indicators", conn, if_exists="replace", index=False)

conn.executescript("""
CREATE VIEW IF NOT EXISTS region_summary AS
SELECT region, year,
  ROUND(AVG(gdp_per_capita),2)    AS avg_gdp,
  ROUND(AVG(poverty_rate)*100,2)  AS avg_poverty_pct,
  ROUND(AVG(gini_index),3)        AS avg_gini,
  ROUND(AVG(hdi),3)               AS avg_hdi,
  ROUND(AVG(life_expectancy),2)   AS avg_life_exp,
  ROUND(AVG(unemployment_rate)*100,2) AS avg_unemployment_pct
FROM indicators GROUP BY region, year;

CREATE VIEW IF NOT EXISTS income_group_trends AS
SELECT income_group, year,
  ROUND(AVG(gdp_per_capita),2)    AS avg_gdp,
  ROUND(AVG(poverty_rate)*100,2)  AS avg_poverty_pct,
  ROUND(AVG(hdi),3)               AS avg_hdi,
  ROUND(AVG(literacy_rate)*100,2) AS avg_literacy_pct,
  ROUND(AVG(internet_access)*100,2) AS avg_internet_pct
FROM indicators GROUP BY income_group, year;

CREATE VIEW IF NOT EXISTS latest_snapshot AS
SELECT * FROM indicators WHERE year = 2023
ORDER BY gdp_per_capita DESC;
""")
conn.commit()
conn.close()
print(f"✅ Generated {len(df):,} records across {len(COUNTRIES)} countries × 9 years")
print(df.head())
