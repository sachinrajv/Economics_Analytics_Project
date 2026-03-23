"""
╔══════════════════════════════════════════════════════════════╗
║   ECONOMICS & POVERTY DATA ANALYTICS PROJECT                ║
║   Stack: SQL (SQLite) + Python                              ║
║   Sections:                                                 ║
║     1. Data Loading & SQL Queries                           ║
║     2. Exploratory Data Analysis                            ║
║     3. Statistical Insights                                 ║
║     4. Data Visualizations (5 dashboards)                   ║
║     5. Predictive Modeling (Poverty Rate Prediction)        ║
╚══════════════════════════════════════════════════════════════╝
"""
import sqlite3, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings("ignore")

OUT = "/home/claude/econ_analytics/outputs"

# ── Palette ────────────────────────────────────────────────────────────────────
DARK   = "#0d1117"; PANEL  = "#161b22"; TEXT   = "#e6edf3"; MUTED  = "#8b949e"
C1     = "#4cc9f0"; C2     = "#f8961e"; C3     = "#06d6a0"; C4     = "#e63946"
C5     = "#9b5de5"; C6     = "#f4d35e"; C7     = "#ee964b"; C8     = "#3a86ff"
REGION_COLORS = {"Africa":C4,"Asia":C1,"Europe":C3,"Middle East":C6,
                 "North America":C5,"Oceania":C7,"South America":C2}
INC_COLORS    = {"high":C3,"middle":C1,"lower-middle":C2,"low":C4}

plt.rcParams.update({
    "figure.facecolor": DARK,"axes.facecolor": PANEL,"axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,"xtick.color": MUTED,"ytick.color": MUTED,
    "text.color": TEXT,"font.family": "DejaVu Sans",
    "axes.grid": True,"grid.color": "#21262d","grid.linewidth": 0.6,
    "legend.facecolor": PANEL,"legend.edgecolor": MUTED,"figure.dpi": 140,
})

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA VIA SQL
# ═══════════════════════════════════════════════════════════════════
print("="*60); print("1. LOADING DATA"); print("="*60)
conn = sqlite3.connect("/home/claude/econ_analytics/data/econ.db")
df   = pd.read_sql("SELECT * FROM indicators", conn)

q_overview  = pd.read_sql("""SELECT income_group,COUNT(DISTINCT country) AS countries,
  ROUND(AVG(gdp_per_capita),0) AS avg_gdp,ROUND(AVG(poverty_rate)*100,1) AS avg_poverty_pct,
  ROUND(AVG(hdi),3) AS avg_hdi,ROUND(AVG(life_expectancy),1) AS avg_life_exp,
  ROUND(AVG(literacy_rate)*100,1) AS avg_literacy_pct
  FROM indicators WHERE year=2023 GROUP BY income_group ORDER BY avg_gdp DESC""", conn)

q_rich = pd.read_sql("SELECT country,region,gdp_per_capita,hdi,poverty_rate,gini_index FROM indicators WHERE year=2023 ORDER BY gdp_per_capita DESC LIMIT 10", conn)
q_poor = pd.read_sql("SELECT country,region,poverty_rate,gdp_per_capita,hdi,life_expectancy FROM indicators WHERE year=2023 ORDER BY poverty_rate DESC LIMIT 10", conn)
q_region= pd.read_sql("SELECT * FROM region_summary", conn)
q_inc   = pd.read_sql("SELECT * FROM income_group_trends", conn)
q_snap  = pd.read_sql("SELECT * FROM latest_snapshot", conn)
q_gini  = pd.read_sql("SELECT country,region,income_group,ROUND(gini_index,3) AS gini,ROUND(gdp_per_capita,0) AS gdp,ROUND(poverty_rate*100,1) AS poverty_pct FROM indicators WHERE year=2023 ORDER BY gini DESC LIMIT 10", conn)
q_hdi   = pd.read_sql("""WITH base AS (SELECT country,hdi AS hdi_2015 FROM indicators WHERE year=2015),
  curr AS (SELECT country,hdi AS hdi_2023 FROM indicators WHERE year=2023)
  SELECT b.country,b.hdi_2015,c.hdi_2023,ROUND(c.hdi_2023-b.hdi_2015,3) AS hdi_change
  FROM base b JOIN curr c ON b.country=c.country ORDER BY hdi_change DESC LIMIT 10""", conn)
conn.close()

print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nIncome Group Overview (2023):\n{q_overview.to_string(index=False)}")
print(f"\nTop 10 Richest Countries (2023):\n{q_rich[['country','gdp_per_capita','hdi','poverty_rate']].to_string(index=False)}")
print(f"\nTop 10 by Poverty Rate (2023):\n{q_poor[['country','poverty_rate','gdp_per_capita','hdi']].to_string(index=False)}")

# ═══════════════════════════════════════════════════════════════════
# 2. EDA
# ═══════════════════════════════════════════════════════════════════
print("\n"+"="*60); print("2. EDA"); print("="*60)
print(df[["gdp_per_capita","poverty_rate","gini_index","hdi","life_expectancy","literacy_rate"]].describe().round(3))

# ═══════════════════════════════════════════════════════════════════
# 3. STATISTICAL INSIGHTS
# ═══════════════════════════════════════════════════════════════════
print("\n"+"="*60); print("3. STATISTICAL TESTS"); print("="*60)

d23 = df[df["year"]==2023]

r1,p1 = stats.pearsonr(d23["gdp_per_capita"], d23["poverty_rate"])
r2,p2 = stats.pearsonr(d23["gdp_per_capita"], d23["hdi"])
r3,p3 = stats.pearsonr(d23["gini_index"],     d23["poverty_rate"])
r4,p4 = stats.pearsonr(d23["edu_exp_pct_gdp"],d23["hdi"])
r5,p5 = stats.pearsonr(d23["internet_access"],d23["poverty_rate"])

print(f"\nPearson Correlations (2023 data):")
print(f"  GDP per capita ↔ Poverty Rate:   r={r1:.4f}, p={p1:.4f}")
print(f"  GDP per capita ↔ HDI:            r={r2:.4f}, p={p2:.4f}")
print(f"  Gini Index     ↔ Poverty Rate:   r={r3:.4f}, p={p3:.4f}")
print(f"  Education Exp  ↔ HDI:            r={r4:.4f}, p={p4:.4f}")
print(f"  Internet Access↔ Poverty Rate:   r={r5:.4f}, p={p5:.4f}")

# ANOVA: HDI across regions
groups_hdi = [g["hdi"].values for _,g in d23.groupby("region")]
f_hdi, p_hdi = stats.f_oneway(*groups_hdi)
print(f"\nANOVA (HDI ~ Region): F={f_hdi:.2f}, p={p_hdi:.6f}")

# COVID t-test: GDP growth 2019 vs 2020
g19 = df[df["year"]==2019]["gdp_growth_pct"]
g20 = df[df["year"]==2020]["gdp_growth_pct"]
t_stat, p_t = stats.ttest_ind(g19, g20)
print(f"T-test GDP growth 2019 vs 2020: t={t_stat:.2f}, p={p_t:.6f}")

# ═══════════════════════════════════════════════════════════════════
# 4. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════
print("\n"+"="*60); print("4. VISUALIZATIONS"); print("="*60)

# ── Fig 1: Overview Dashboard ──────────────────────────────────────
fig = plt.figure(figsize=(22,15))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.45,wspace=0.38)

# GDP by income group bar
ax1 = fig.add_subplot(gs[0,0])
grps = q_overview.sort_values("avg_gdp",ascending=True)
cols = [INC_COLORS[g] for g in grps["income_group"]]
ax1.barh(grps["income_group"],grps["avg_gdp"],color=cols,alpha=0.88)
ax1.set_title("Avg GDP/Capita by Income Group\n(2023)",color=TEXT,fontweight="bold",fontsize=11)
ax1.set_xlabel("USD")
for i,(v,g) in enumerate(zip(grps["avg_gdp"],grps["income_group"])):
    ax1.text(v+500,i,f"${v:,.0f}",va="center",color=TEXT,fontsize=8)

# HDI by region
ax2 = fig.add_subplot(gs[0,1])
hdi_r = d23.groupby("region")["hdi"].mean().sort_values()
cols2 = [REGION_COLORS[r] for r in hdi_r.index]
ax2.barh(hdi_r.index,hdi_r.values,color=cols2,alpha=0.88)
ax2.axvline(0.7,color=TEXT,linestyle="--",linewidth=1,alpha=0.6,label="0.70 threshold")
ax2.set_title("Avg HDI by Region (2023)",color=TEXT,fontweight="bold",fontsize=11)
ax2.set_xlabel("HDI Score"); ax2.legend(fontsize=7)

# Poverty trend by income group
ax3 = fig.add_subplot(gs[0,2])
for inc,col in INC_COLORS.items():
    sub = q_inc[q_inc["income_group"]==inc]
    ax3.plot(sub["year"],sub["avg_poverty_pct"],color=col,marker="o",markersize=4,
             linewidth=2,label=inc)
ax3.axvspan(2019.5,2021,alpha=0.1,color=C4,label="COVID period")
ax3.set_title("Poverty Rate Trend\nby Income Group",color=TEXT,fontweight="bold",fontsize=11)
ax3.set_xlabel("Year"); ax3.set_ylabel("Avg Poverty %"); ax3.legend(fontsize=7)

# GDP vs Poverty scatter
ax4 = fig.add_subplot(gs[1,0])
scatter = ax4.scatter(np.log1p(d23["gdp_per_capita"]),d23["poverty_rate"]*100,
                      c=[list(REGION_COLORS.keys()).index(r) for r in d23["region"]],
                      cmap="tab10",alpha=0.7,s=50)
ax4.set_title("GDP per Capita vs Poverty Rate\n(log scale, 2023)",color=TEXT,fontweight="bold",fontsize=11)
ax4.set_xlabel("Log GDP per Capita"); ax4.set_ylabel("Poverty Rate %")

# Gini by region boxplot
ax5 = fig.add_subplot(gs[1,1])
regions = sorted(d23["region"].unique())
gini_data = [d23[d23["region"]==r]["gini_index"].values for r in regions]
bp = ax5.boxplot(gini_data,labels=[r[:8] for r in regions],patch_artist=True,
                 medianprops=dict(color=TEXT,linewidth=2))
for patch,col in zip(bp["boxes"],[REGION_COLORS[r] for r in regions]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax5.set_title("Gini Index Distribution\nby Region (2023)",color=TEXT,fontweight="bold",fontsize=11)
ax5.set_ylabel("Gini Index"); ax5.tick_params(axis="x",rotation=30,labelsize=7)

# Internet access trend
ax6 = fig.add_subplot(gs[1,2])
for inc,col in INC_COLORS.items():
    sub = q_inc[q_inc["income_group"]==inc]
    ax6.plot(sub["year"],sub["avg_internet_pct"],color=col,marker="s",markersize=4,
             linewidth=2,label=inc)
ax6.set_title("Internet Access Trend\nby Income Group (%)",color=TEXT,fontweight="bold",fontsize=11)
ax6.set_xlabel("Year"); ax6.set_ylabel("Internet Access %"); ax6.legend(fontsize=7)

# Life expectancy vs HDI scatter
ax7 = fig.add_subplot(gs[2,0])
ax7.scatter(d23["life_expectancy"],d23["hdi"],
            c=[list(REGION_COLORS.values())[list(REGION_COLORS.keys()).index(r)] for r in d23["region"]],
            alpha=0.75,s=55)
ax7.set_title("Life Expectancy vs HDI (2023)",color=TEXT,fontweight="bold",fontsize=11)
ax7.set_xlabel("Life Expectancy (years)"); ax7.set_ylabel("HDI")

# GDP growth: COVID bar chart
ax8 = fig.add_subplot(gs[2,1])
gdp_yoy = df.groupby("year")["gdp_growth_pct"].mean()
bar_c = [C4 if y==2020 else C3 if y==2021 else C1 for y in gdp_yoy.index]
ax8.bar(gdp_yoy.index,gdp_yoy.values,color=bar_c,alpha=0.88)
ax8.axhline(0,color=MUTED,linewidth=1)
ax8.set_title("Avg Global GDP Growth\nper Year",color=TEXT,fontweight="bold",fontsize=11)
ax8.set_xlabel("Year"); ax8.set_ylabel("Growth %")
ax8.text(2020,-1.8,"COVID\nImpact",color=C4,fontsize=8,ha="center")

# Literacy vs Poverty
ax9 = fig.add_subplot(gs[2,2])
ax9.scatter(d23["literacy_rate"]*100,d23["poverty_rate"]*100,
            c=d23["hdi"],cmap="RdYlGn",s=55,alpha=0.8)
ax9.set_title("Literacy Rate vs Poverty Rate\n(coloured by HDI, 2023)",color=TEXT,fontweight="bold",fontsize=11)
ax9.set_xlabel("Literacy Rate %"); ax9.set_ylabel("Poverty Rate %")

fig.suptitle("🌍  ECONOMICS & POVERTY ANALYTICS DASHBOARD",color=TEXT,fontsize=18,fontweight="bold",y=1.01)
plt.savefig(f"{OUT}/01_overview_dashboard.png",bbox_inches="tight",facecolor=DARK,dpi=140)
plt.close(); print("✅ 01_overview_dashboard.png")

# ── Fig 2: Regional Deep Dive ───────────────────────────────────────
fig,axes = plt.subplots(2,2,figsize=(18,11))
fig.patch.set_facecolor(DARK)
for ax in axes.flat: ax.set_facecolor(PANEL)

for region,col in REGION_COLORS.items():
    sub = q_region[q_region["region"]==region]
    axes[0,0].plot(sub["year"],sub["avg_gdp"],color=col,linewidth=2,marker="o",markersize=4,label=region)
axes[0,0].axvspan(2019.5,2021,alpha=0.1,color=C4)
axes[0,0].set_title("GDP per Capita by Region",color=TEXT,fontweight="bold")
axes[0,0].set_xlabel("Year"); axes[0,0].set_ylabel("Avg GDP per Capita (USD)")
axes[0,0].legend(fontsize=7,ncol=2)

for region,col in REGION_COLORS.items():
    sub = q_region[q_region["region"]==region]
    axes[0,1].plot(sub["year"],sub["avg_poverty_pct"],color=col,linewidth=2,marker="s",markersize=4,label=region)
axes[0,1].axvspan(2019.5,2021,alpha=0.1,color=C4)
axes[0,1].set_title("Poverty Rate Trend by Region",color=TEXT,fontweight="bold")
axes[0,1].set_xlabel("Year"); axes[0,1].set_ylabel("Avg Poverty %"); axes[0,1].legend(fontsize=7,ncol=2)

for region,col in REGION_COLORS.items():
    sub = q_region[q_region["region"]==region]
    axes[1,0].plot(sub["year"],sub["avg_hdi"],color=col,linewidth=2,marker="^",markersize=4,label=region)
axes[1,0].set_title("HDI Trend by Region",color=TEXT,fontweight="bold")
axes[1,0].set_xlabel("Year"); axes[1,0].set_ylabel("Avg HDI"); axes[1,0].legend(fontsize=7,ncol=2)

snap23 = d23.sort_values("unemployment_rate",ascending=False).head(15)
axes[1,1].barh(snap23["country"],snap23["unemployment_rate"]*100,
               color=[REGION_COLORS[r] for r in snap23["region"]],alpha=0.85)
axes[1,1].set_title("Top 15 Countries by Unemployment\n(2023)",color=TEXT,fontweight="bold")
axes[1,1].set_xlabel("Unemployment Rate %")

fig.suptitle("Regional Economics Deep Dive",color=TEXT,fontsize=16,fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/02_regional_deepdive.png",bbox_inches="tight",facecolor=DARK,dpi=140)
plt.close(); print("✅ 02_regional_deepdive.png")

# ── Fig 3: Correlation Heatmap & Top Countries ──────────────────────
fig,axes = plt.subplots(1,2,figsize=(18,8))
fig.patch.set_facecolor(DARK)
for ax in axes: ax.set_facecolor(PANEL)

numeric_cols = ["gdp_per_capita","poverty_rate","gini_index","unemployment_rate",
                "literacy_rate","life_expectancy","hdi","internet_access",
                "health_exp_pct_gdp","edu_exp_pct_gdp","inflation_rate"]
corr = d23[numeric_cols].corr()
mask = np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(corr,ax=axes[0],mask=mask,cmap="RdYlGn",center=0,annot=True,
            fmt=".2f",annot_kws={"size":7},linewidths=0.5,
            cbar_kws={"shrink":0.7})
axes[0].set_title("Correlation Matrix — Socioeconomic Indicators (2023)",
                  color=TEXT,fontweight="bold",fontsize=12)
axes[0].tick_params(axis="both",labelsize=7)

# Top & Bottom HDI countries
top_hdi    = d23.nlargest(10,"hdi")[["country","hdi","gdp_per_capita"]]
bottom_hdi = d23.nsmallest(10,"hdi")[["country","hdi","gdp_per_capita"]]
all_hdi    = pd.concat([top_hdi,bottom_hdi]).reset_index(drop=True)
col_bars   = [C3]*10 + [C4]*10
axes[1].barh(all_hdi["country"][::-1],all_hdi["hdi"][::-1],color=col_bars[::-1],alpha=0.85)
axes[1].axvline(0.7,color=TEXT,linestyle="--",linewidth=1,alpha=0.6)
axes[1].set_title("Top & Bottom 10 Countries by HDI (2023)",color=TEXT,fontweight="bold",fontsize=12)
axes[1].set_xlabel("HDI Score")

plt.tight_layout()
plt.savefig(f"{OUT}/03_correlation_countries.png",bbox_inches="tight",facecolor=DARK,dpi=140)
plt.close(); print("✅ 03_correlation_countries.png")

# ── Fig 4: COVID Impact ─────────────────────────────────────────────
fig,axes = plt.subplots(1,3,figsize=(20,7))
fig.patch.set_facecolor(DARK)
for ax in axes: ax.set_facecolor(PANEL)

# GDP growth distribution 2019–2022
for yr,col in zip([2019,2020,2021,2022],[C3,C4,C1,C2]):
    vals = df[df["year"]==yr]["gdp_growth_pct"]
    axes[0].hist(vals,bins=15,alpha=0.6,color=col,label=str(yr))
axes[0].axvline(0,color=TEXT,linewidth=1.5,linestyle="--")
axes[0].set_title("GDP Growth Distribution\n2019–2022",color=TEXT,fontweight="bold")
axes[0].set_xlabel("GDP Growth %"); axes[0].legend()

# Unemployment 2019 vs 2020 vs 2022 by region
years_un = [2019,2020,2022]
x = np.arange(len(regions)); w = 0.25
for i,(yr,col) in enumerate(zip(years_un,[C3,C4,C1])):
    vals = [df[(df["year"]==yr)&(df["region"]==r)]["unemployment_rate"].mean()*100 for r in regions]
    axes[1].bar(x+i*w,vals,width=w,color=col,alpha=0.85,label=str(yr))
axes[1].set_xticks(x+w); axes[1].set_xticklabels([r[:7] for r in regions],rotation=30,fontsize=7)
axes[1].set_title("Unemployment by Region\n2019 / 2020 / 2022",color=TEXT,fontweight="bold")
axes[1].set_ylabel("Unemployment %"); axes[1].legend()

# Poverty rate 2019 vs 2021 vs 2023
years_p = [2019,2020,2021,2023]
x = np.arange(len(regions)); w = 0.2
for i,(yr,col) in enumerate(zip(years_p,[C3,C4,C1,C2])):
    vals = [df[(df["year"]==yr)&(df["region"]==r)]["poverty_rate"].mean()*100 for r in regions]
    axes[2].bar(x+i*w,vals,width=w,color=col,alpha=0.85,label=str(yr))
axes[2].set_xticks(x+1.5*w); axes[2].set_xticklabels([r[:7] for r in regions],rotation=30,fontsize=7)
axes[2].set_title("Poverty Rate by Region\n2019/2020/2021/2023",color=TEXT,fontweight="bold")
axes[2].set_ylabel("Avg Poverty %"); axes[2].legend()

fig.suptitle("COVID-19 Economic Impact Analysis",color=TEXT,fontsize=16,fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/04_covid_impact.png",bbox_inches="tight",facecolor=DARK,dpi=140)
plt.close(); print("✅ 04_covid_impact.png")

# ═══════════════════════════════════════════════════════════════════
# 5. PREDICTIVE MODELING — Poverty Rate Regression
# ═══════════════════════════════════════════════════════════════════
print("\n"+"="*60); print("5. PREDICTIVE MODELING"); print("="*60)

le = LabelEncoder()
df["region_enc"]    = le.fit_transform(df["region"])
df["inc_enc"]       = le.fit_transform(df["income_group"])

FEATURES = ["gdp_per_capita","gini_index","unemployment_rate","literacy_rate",
            "life_expectancy","hdi","internet_access","health_exp_pct_gdp",
            "edu_exp_pct_gdp","inflation_rate","region_enc","inc_enc","year"]
TARGET = "poverty_rate"

X = df[FEATURES]; y = df[TARGET]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
Xs_tr  = scaler.fit_transform(X_train); Xs_te = scaler.transform(X_test)

models = {
    "Ridge Regression":      Ridge(alpha=1.0),
    "Random Forest":         RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1),
    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=100,random_state=42),
}
results = {}
for name,model in models.items():
    Xf = Xs_tr if "Ridge" in name else X_train
    Xe = Xs_te if "Ridge" in name else X_test
    model.fit(Xf,y_train); y_pred = model.predict(Xe)
    r2  = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    cv  = cross_val_score(model,X if "Ridge" not in name else scaler.fit_transform(X),
                          y,cv=5,scoring="r2",n_jobs=-1).mean()
    results[name]={"model":model,"y_pred":y_pred,"r2":r2,"mae":mae,"cv":cv}
    print(f"\n── {name}: R²={r2:.4f} | MAE={mae:.4f} | CV R²={cv:.4f}")

best_name = max(results,key=lambda n: results[n]["r2"])
best_res  = results[best_name]

# ── Fig 5: Model Results ────────────────────────────────────────────
fig,axes = plt.subplots(1,3,figsize=(20,7))
fig.patch.set_facecolor(DARK)
for ax in axes: ax.set_facecolor(PANEL)

mc = [C1,C3,C2]
for (name,res),col in zip(results.items(),mc):
    axes[0].scatter([name],[res["r2"]],color=col,s=160,zorder=3,label=f"R²={res['r2']:.3f}")
    axes[0].bar([name],[res["r2"]],color=col,alpha=0.4,width=0.5)
axes[0].set_title("Model R² Comparison\n(Poverty Rate Prediction)",color=TEXT,fontweight="bold")
axes[0].set_ylabel("R² Score"); axes[0].set_ylim(0,1); axes[0].legend(fontsize=8)

axes[1].scatter(y_test,best_res["y_pred"],color=C1,alpha=0.6,s=30)
mn,mx = min(y_test.min(),best_res["y_pred"].min()),max(y_test.max(),best_res["y_pred"].max())
axes[1].plot([mn,mx],[mn,mx],color=C4,linewidth=2,linestyle="--",label="Perfect fit")
axes[1].set_title(f"Actual vs Predicted — {best_name}",color=TEXT,fontweight="bold")
axes[1].set_xlabel("Actual Poverty Rate"); axes[1].set_ylabel("Predicted"); axes[1].legend()

rf = results["Random Forest"]["model"]
fi = pd.Series(rf.feature_importances_,index=FEATURES).sort_values(ascending=True)
fi.plot(kind="barh",ax=axes[2],color=C2,alpha=0.85)
axes[2].set_title("Feature Importance\n(Random Forest)",color=TEXT,fontweight="bold")
axes[2].set_xlabel("Importance Score")

fig.suptitle("Poverty Rate Predictive Model Results",color=TEXT,fontsize=16,fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/05_model_results.png",bbox_inches="tight",facecolor=DARK,dpi=140)
plt.close(); print(f"\n✅ 05_model_results.png")

# ── SUMMARY ────────────────────────────────────────────────────────
print(f"""
{'='*60}
PROJECT SUMMARY
{'='*60}
📊 Dataset     : {len(df):,} records — 50 countries × 9 years (2015–2023)
🗄️  SQL Queries : 10 analytical queries + 3 database views
🔍 EDA         : 17 socioeconomic indicators analysed
📈 Statistics  : Pearson (5 pairs), ANOVA, T-Test
🤖 Best Model  : {best_name} (R²={results[best_name]['r2']:.4f})
📁 Outputs     : 5 dashboard PNGs

KEY FINDINGS:
  • Strong negative correlation: GDP ↔ Poverty (r={r1:.3f})
  • High positive correlation:   GDP ↔ HDI     (r={r2:.3f})
  • Internet access is a strong proxy for development
  • COVID-19 caused the sharpest single-year GDP drop in the dataset
  • Africa has highest avg poverty rate; Europe has lowest
  • {best_name} explains {results[best_name]['r2']*100:.1f}% of poverty rate variance
""")
