-- ============================================================
-- ECONOMICS & POVERTY ANALYTICS — SQL QUERIES
-- Database: econ.db  |  Main table: indicators
-- ============================================================

-- 1. GLOBAL OVERVIEW: Average indicators by income group (2023)
SELECT income_group,
       COUNT(DISTINCT country) AS countries,
       ROUND(AVG(gdp_per_capita),0)       AS avg_gdp_per_capita,
       ROUND(AVG(poverty_rate)*100,1)     AS avg_poverty_pct,
       ROUND(AVG(hdi),3)                  AS avg_hdi,
       ROUND(AVG(life_expectancy),1)      AS avg_life_exp,
       ROUND(AVG(literacy_rate)*100,1)    AS avg_literacy_pct
FROM indicators WHERE year=2023
GROUP BY income_group
ORDER BY avg_gdp_per_capita DESC;

-- 2. TOP 10 RICHEST COUNTRIES (2023)
SELECT country, region, gdp_per_capita, hdi, poverty_rate, gini_index
FROM indicators WHERE year=2023
ORDER BY gdp_per_capita DESC LIMIT 10;

-- 3. TOP 10 POOREST COUNTRIES (2023, by poverty rate)
SELECT country, region, poverty_rate, gdp_per_capita, hdi, life_expectancy
FROM indicators WHERE year=2023
ORDER BY poverty_rate DESC LIMIT 10;

-- 4. GDP GROWTH: COVID impact (2019 vs 2020 vs 2021)
SELECT country, region,
  ROUND(AVG(CASE WHEN year=2019 THEN gdp_growth_pct END),2) AS growth_2019,
  ROUND(AVG(CASE WHEN year=2020 THEN gdp_growth_pct END),2) AS growth_2020,
  ROUND(AVG(CASE WHEN year=2021 THEN gdp_growth_pct END),2) AS growth_2021
FROM indicators GROUP BY country, region
ORDER BY growth_2020 ASC LIMIT 15;

-- 5. INEQUALITY: Highest Gini Index (2023)
SELECT country, region, income_group,
       ROUND(gini_index,3) AS gini,
       ROUND(gdp_per_capita,0) AS gdp,
       ROUND(poverty_rate*100,1) AS poverty_pct
FROM indicators WHERE year=2023
ORDER BY gini DESC LIMIT 10;

-- 6. REGIONAL TRENDS: Poverty reduction over time
SELECT region, year,
       ROUND(AVG(poverty_rate)*100,2) AS avg_poverty_pct,
       ROUND(AVG(gdp_per_capita),0)   AS avg_gdp
FROM indicators GROUP BY region, year ORDER BY region, year;

-- 7. HDI PROGRESS: Countries with fastest improvement (2015 vs 2023)
WITH base AS (SELECT country, hdi AS hdi_2015 FROM indicators WHERE year=2015),
     curr AS (SELECT country, hdi AS hdi_2023 FROM indicators WHERE year=2023)
SELECT b.country,
       b.hdi_2015, c.hdi_2023,
       ROUND(c.hdi_2023 - b.hdi_2015, 3) AS hdi_change
FROM base b JOIN curr c ON b.country=c.country
ORDER BY hdi_change DESC LIMIT 10;

-- 8. INTERNET ACCESS vs POVERTY CORRELATION
SELECT country, year,
       ROUND(internet_access*100,1) AS internet_pct,
       ROUND(poverty_rate*100,1)    AS poverty_pct,
       ROUND(gdp_per_capita,0)      AS gdp
FROM indicators WHERE year=2023
ORDER BY internet_pct DESC;

-- 9. UNEMPLOYMENT SPIKE: COVID impact by region
SELECT region,
  ROUND(AVG(CASE WHEN year=2019 THEN unemployment_rate END)*100,2) AS unemp_2019,
  ROUND(AVG(CASE WHEN year=2020 THEN unemployment_rate END)*100,2) AS unemp_2020,
  ROUND(AVG(CASE WHEN year=2021 THEN unemployment_rate END)*100,2) AS unemp_2021,
  ROUND(AVG(CASE WHEN year=2022 THEN unemployment_rate END)*100,2) AS unemp_2022
FROM indicators GROUP BY region;

-- 10. EDUCATION vs HDI LINK
SELECT income_group,
       ROUND(AVG(edu_exp_pct_gdp),2)  AS avg_edu_spend_pct,
       ROUND(AVG(literacy_rate)*100,1) AS avg_literacy,
       ROUND(AVG(hdi),3)               AS avg_hdi
FROM indicators WHERE year=2023
GROUP BY income_group ORDER BY avg_hdi DESC;
