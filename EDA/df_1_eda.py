import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Descriptive Analysis

# Style
plt.rcParams.update({
    'figure.facecolor': '#0F1117',
    'axes.facecolor':   '#1A1D27',
    'axes.edgecolor':   '#2E3145',
    'axes.labelcolor':  '#C8CCDF',
    'axes.titlecolor':  '#FFFFFF',
    'xtick.color':      '#7A7F99',
    'ytick.color':      '#7A7F99',
    'text.color':       '#C8CCDF',
    'grid.color':       '#2E3145',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'font.family':      'monospace',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

ACCENT   = ['#4F8EF7', '#F7784F', '#4FF7A0', '#F7D84F', '#BF4FF7', '#4FF7F0']
BG_DARK  = '#0F1117'
BG_MID   = '#1A1D27'
TEXT_HI  = '#FFFFFF'
TEXT_LO  = '#7A7F99'

def fmt_vnd(x, _=None):
    if abs(x) >= 1e9:  return f'{x/1e9:.1f}B'
    if abs(x) >= 1e6:  return f'{x/1e6:.1f}M'
    return f'{x:,.0f}'

def section_title(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("df_1.csv", parse_dates=['order_date', 'signup_date'])

# ── Feature engineering ────────────────────────────────────────────────────────
df = df[df['order_status'] != 'cancelled'].copy()
df['revenue']      = df['unit_price'] * df['quantity'] - df['discount_amount']
df['gross_profit'] = (df['price'] - df['cogs']) / df['price'] * df['revenue']
df['margin']       = (df['price'] - df['cogs']) / df['price']
df['year_month']   = df['order_date'].dt.to_period('M')
df['year']         = df['order_date'].dt.year


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Revenue Breakdown (4 dimensions)
# ══════════════════════════════════════════════════════════════════════════════
section_title("FIGURE 1 · Revenue Breakdown")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle('REVENUE BREAKDOWN', fontsize=22, fontweight='bold',
             color=TEXT_HI, y=0.98)

dims = [
    ('category',            'By Category'),
    ('segment',             'By Segment'),
    ('region',              'By Region'),
    ('acquisition_channel', 'By Acquisition Channel'),
]

for ax, (col, title) in zip(axes.flatten(), dims):
    data = df.groupby(col)['revenue'].sum().sort_values(ascending=True)
    bars = ax.barh(data.index, data.values,
                   color=ACCENT[:len(data)], edgecolor='none', height=0.6)
    for bar, val in zip(bars, data.values):
        ax.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                fmt_vnd(val), va='center', fontsize=9, color=TEXT_HI)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
    ax.set_facecolor(BG_MID)
    ax.tick_params(labelsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('fig1_revenue_breakdown.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.show()
print("✅ fig1_revenue_breakdown.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Revenue Trend Over Time
# ══════════════════════════════════════════════════════════════════════════════
section_title("FIGURE 2 · Revenue Trend")

monthly = df.groupby('year_month')['revenue'].sum().reset_index()
monthly['year_month_dt'] = monthly['year_month'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(18, 5))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_MID)

ax.fill_between(monthly['year_month_dt'], monthly['revenue'],
                alpha=0.18, color=ACCENT[0])
ax.plot(monthly['year_month_dt'], monthly['revenue'],
        color=ACCENT[0], linewidth=2.2)

# Rolling 3-month avg
monthly['rolling'] = monthly['revenue'].rolling(3, min_periods=1).mean()
ax.plot(monthly['year_month_dt'], monthly['rolling'],
        color=ACCENT[1], linewidth=1.5, linestyle='--', label='3M Rolling Avg')

ax.set_title('MONTHLY REVENUE TREND', fontsize=16, fontweight='bold', color=TEXT_HI)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig2_revenue_trend.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.show()
print("✅ fig2_revenue_trend.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Profitability: Margin by Category & Segment
# ══════════════════════════════════════════════════════════════════════════════
section_title("FIGURE 3 · Margin Analysis")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle('GROSS MARGIN ANALYSIS', fontsize=18, fontweight='bold',
             color=TEXT_HI, y=1.01)

for ax, col, title in zip(axes,
                           ['category', 'segment'],
                           ['Avg Gross Margin by Category', 'Avg Gross Margin by Segment']):
    data = df.groupby(col)['margin'].mean().sort_values(ascending=True)
    colors = [ACCENT[2] if v >= data.median() else ACCENT[1] for v in data.values]
    bars = ax.barh(data.index, data.values * 100, color=colors, edgecolor='none', height=0.55)
    for bar, val in zip(bars, data.values):
        ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=9, color=TEXT_HI)
    ax.axvline(data.median() * 100, color=TEXT_LO, linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Gross Margin (%)')
    ax.set_facecolor(BG_MID)

plt.tight_layout()
plt.savefig('fig3_margin_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.show()
print("✅ fig3_margin_analysis.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — BCG-style Bubble: Revenue vs Margin by Product
# ══════════════════════════════════════════════════════════════════════════════
section_title("FIGURE 4 · Revenue vs Margin (Product Quadrant)")

prod = df.groupby('product_name').agg(
    revenue      = ('revenue',      'sum'),
    margin       = ('margin',       'mean'),
    quantity_sold= ('quantity',     'sum'),
).reset_index()

med_rev = prod['revenue'].median()
med_mar = prod['margin'].median()

def quadrant(row):
    if row['revenue'] >= med_rev and row['margin'] >= med_mar: return '🌟 Star'
    if row['revenue'] >= med_rev and row['margin'] <  med_mar: return '⚠️  High Rev / Low Margin'
    if row['revenue'] <  med_rev and row['margin'] >= med_mar: return '💎 High Margin / Low Rev'
    return '🐌 Low Both'

prod['quadrant'] = prod.apply(quadrant, axis=1)

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_MID)

qcolors = {
    '🌟 Star':                  ACCENT[2],
    '⚠️  High Rev / Low Margin': ACCENT[1],
    '💎 High Margin / Low Rev':  ACCENT[0],
    '🐌 Low Both':               TEXT_LO,
}

for q, grp in prod.groupby('quadrant'):
    ax.scatter(grp['revenue'], grp['margin'] * 100,
               s=grp['quantity_sold'] / grp['quantity_sold'].max() * 600 + 40,
               color=qcolors[q], alpha=0.75, edgecolors='none', label=q)

# Quadrant lines
ax.axvline(med_rev, color=TEXT_LO, linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(med_mar * 100, color=TEXT_LO, linestyle='--', linewidth=1, alpha=0.5)

# Label top-10 by revenue
top10 = prod.nlargest(10, 'revenue')
for _, row in top10.iterrows():
    ax.annotate(row['product_name'][:18],
                (row['revenue'], row['margin']*100),
                fontsize=7, color=TEXT_HI, alpha=0.85,
                xytext=(5, 3), textcoords='offset points')

ax.set_xlabel('Total Revenue', fontsize=11)
ax.set_ylabel('Avg Gross Margin (%)', fontsize=11)
ax.set_title('PRODUCT QUADRANT  ·  bubble size = quantity sold',
             fontsize=15, fontweight='bold', color=TEXT_HI)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
ax.legend(fontsize=9, framealpha=0.2)

# Quadrant labels
ax.text(prod['revenue'].max()*0.98, med_mar*100*1.02, 'HIGH REV\nLOW MARGIN',
        ha='right', fontsize=8, color=ACCENT[1], alpha=0.5)
ax.text(prod['revenue'].max()*0.98, prod['margin'].max()*100*0.97, 'STAR',
        ha='right', fontsize=8, color=ACCENT[2], alpha=0.5)

plt.tight_layout()
plt.savefig('fig4_product_quadrant.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.show()
print("✅ fig4_product_quadrant.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Segment: Revenue vs Margin (Risk Matrix)
# ══════════════════════════════════════════════════════════════════════════════
section_title("FIGURE 5 · Segment Risk Matrix (Revenue vs Margin)")

seg = df.groupby('segment').agg(
    revenue= ('revenue', 'sum'),
    margin = ('margin',  'mean'),
).reset_index()

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_MID)

colors = [ACCENT[1] if m < seg['margin'].median() else ACCENT[2]
          for m in seg['margin']]

scatter = ax.scatter(seg['revenue'], seg['margin']*100,
                     s=seg['revenue']/seg['revenue'].max()*1200+100,
                     c=colors, alpha=0.85, edgecolors='white', linewidths=0.5)

for _, row in seg.iterrows():
    ax.annotate(row['segment'],
                (row['revenue'], row['margin']*100),
                fontsize=10, fontweight='bold', color=TEXT_HI,
                xytext=(8, 4), textcoords='offset points')

ax.axvline(seg['revenue'].median(), color=TEXT_LO, linestyle='--', alpha=0.4)
ax.axhline(seg['margin'].median()*100, color=TEXT_LO, linestyle='--', alpha=0.4)

ax.set_xlabel('Total Revenue', fontsize=11)
ax.set_ylabel('Avg Gross Margin (%)', fontsize=11)
ax.set_title('SEGMENT RISK MATRIX  ·  High Revenue + Low Margin = Danger Zone',
             fontsize=14, fontweight='bold', color=TEXT_HI)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))

plt.tight_layout()
plt.savefig('fig5_segment_risk.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.show()
print("✅ fig5_segment_risk.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Region: Revenue + Margin heatmap-style bar
# ══════════════════════════════════════════════════════════════════════════════
section_title("FIGURE 6 · Region Performance")

reg = df.groupby('region').agg(
    revenue= ('revenue', 'sum'),
    margin = ('margin',  'mean'),
    orders = ('order_id','nunique'),
).reset_index().sort_values('revenue', ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle('REGION PERFORMANCE', fontsize=18, fontweight='bold', color=TEXT_HI)

for ax, (col, label, color) in zip(axes, [
    ('revenue', 'Total Revenue',     ACCENT[0]),
    ('margin',  'Avg Gross Margin',  ACCENT[2]),
    ('orders',  'Unique Orders',     ACCENT[3]),
]):
    data = reg.set_index('region')[col].sort_values(ascending=True)
    bars = ax.barh(data.index, data.values, color=color, alpha=0.85, height=0.55)
    for bar, val in zip(bars, data.values):
        lbl = f'{val:.1%}' if col == 'margin' else fmt_vnd(val)
        ax.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                lbl, va='center', fontsize=9, color=TEXT_HI)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_facecolor(BG_MID)
    if col != 'margin':
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))

plt.tight_layout()
plt.savefig('fig6_region_performance.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.show()
print("✅ fig6_region_performance.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Acquisition Channel: Revenue + Margin + Orders
# ══════════════════════════════════════════════════════════════════════════════
section_title("FIGURE 7 · Acquisition Channel Effectiveness")

ch = df.groupby('acquisition_channel').agg(
    revenue      = ('revenue',  'sum'),
    margin       = ('margin',   'mean'),
    orders       = ('order_id', 'nunique'),
    customers    = ('customer_id','nunique'),
).reset_index()
ch['rev_per_customer'] = ch['revenue'] / ch['customers']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle('ACQUISITION CHANNEL ANALYSIS', fontsize=18, fontweight='bold', color=TEXT_HI)

plots = [
    ('revenue',          'Total Revenue',          ACCENT[0]),
    ('margin',           'Avg Gross Margin',        ACCENT[2]),
    ('customers',        'Unique Customers',        ACCENT[3]),
    ('rev_per_customer', 'Revenue per Customer',    ACCENT[4]),
]

for ax, (col, title, color) in zip(axes.flatten(), plots):
    data = ch.set_index('acquisition_channel')[col].sort_values(ascending=True)
    bars = ax.barh(data.index, data.values, color=color, alpha=0.85, height=0.5)
    for bar, val in zip(bars, data.values):
        lbl = f'{val:.1%}' if col == 'margin' else fmt_vnd(val)
        ax.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                lbl, va='center', fontsize=9, color=TEXT_HI)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_facecolor(BG_MID)
    if col != 'margin':
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))

plt.tight_layout()
plt.savefig('fig7_channel_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.show()
print("✅ fig7_channel_analysis.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Top 10 Products: High Revenue Low Margin vs High Margin Low Revenue
# ══════════════════════════════════════════════════════════════════════════════
section_title("FIGURE 8 · Product Anomalies")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle('PRODUCT ANOMALIES', fontsize=18, fontweight='bold', color=TEXT_HI)

# High revenue low margin
danger = prod.nlargest(20, 'revenue').nsmallest(10, 'margin')[['product_name','revenue','margin']]
ax = axes[0]
ax.set_facecolor(BG_MID)
bars = ax.barh(danger['product_name'].str[:20], danger['margin']*100,
               color=ACCENT[1], alpha=0.85, height=0.55)
for bar, val in zip(bars, danger['margin'].values):
    ax.text(val*100 + 0.2, bar.get_y()+bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=8, color=TEXT_HI)
ax.set_title('⚠️  High Revenue · Low Margin', fontsize=12, fontweight='bold', color=ACCENT[1])
ax.set_xlabel('Gross Margin (%)')

# High margin low revenue
hidden = prod.nlargest(20, 'margin').nsmallest(10, 'revenue')[['product_name','revenue','margin']]
ax = axes[1]
ax.set_facecolor(BG_MID)
bars = ax.barh(hidden['product_name'].str[:20], hidden['revenue'],
               color=ACCENT[2], alpha=0.85, height=0.55)
for bar, val in zip(bars, hidden['revenue'].values):
    ax.text(val + val*0.01, bar.get_y()+bar.get_height()/2,
            fmt_vnd(val), va='center', fontsize=8, color=TEXT_HI)
ax.set_title('💎 High Margin · Low Revenue', fontsize=12, fontweight='bold', color=ACCENT[2])
ax.set_xlabel('Total Revenue')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))

plt.tight_layout()
plt.savefig('fig8_product_anomalies.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.show()
print("✅ fig8_product_anomalies.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
section_title("EXECUTIVE SUMMARY")

total_rev    = df['revenue'].sum()
total_profit = df['gross_profit'].sum()
avg_margin   = df['margin'].mean()
top_cat      = df.groupby('category')['revenue'].sum().idxmax()
top_region   = df.groupby('region')['revenue'].sum().idxmax()
top_channel  = df.groupby('acquisition_channel')['revenue'].sum().idxmax()
danger_seg   = df.groupby('segment').agg(r=('revenue','sum'),m=('margin','mean'))
danger_seg   = danger_seg[danger_seg['r'] > danger_seg['r'].median()].nsmallest(1,'m').index[0]

print(f"""
  Total Revenue     : {fmt_vnd(total_rev)}
  Total Gross Profit: {fmt_vnd(total_profit)}
  Avg Gross Margin  : {avg_margin:.1%}
  ─────────────────────────────────────
  Top Category      : {top_cat}
  Top Region        : {top_region}  → focus expansion here
  Top Channel       : {top_channel}
  Danger Segment    : {danger_seg}  → high revenue, low margin
""")

# Diagnostic Analysis

# Style 
plt.rcParams.update({
    'figure.facecolor': '#0B0E1A',
    'axes.facecolor':   '#13172A',
    'axes.edgecolor':   '#252A45',
    'axes.labelcolor':  '#B8BDD6',
    'axes.titlecolor':  '#FFFFFF',
    'xtick.color':      '#6B7099',
    'ytick.color':      '#6B7099',
    'text.color':       '#B8BDD6',
    'grid.color':       '#252A45',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

BG       = '#0B0E1A'
BG_MID   = '#13172A'
BG_CARD  = '#1C2140'
C_BLUE   = '#4F8EF7'
C_RED    = '#F75A5A'
C_GREEN  = '#4FF7A0'
C_AMBER  = '#F7C94F'
C_PURPLE = '#BF4FF7'
C_TEAL   = '#4FF7F0'
TEXT_HI  = '#FFFFFF'
TEXT_LO  = '#6B7099'

def fmt_b(x, _=None):
    if abs(x) >= 1e9: return f'{x/1e9:.2f}B'
    if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
    if abs(x) >= 1e3: return f'{x/1e3:.0f}K'
    return f'{x:.2f}'

def section(t): print(f"\n{'═'*60}\n  {t}\n{'═'*60}")

# Load 
df        = pd.read_csv("df_1.csv", parse_dates=['order_date'])
orders    = pd.read_csv("orders.csv", parse_dates=['order_date'])
returns   = pd.read_csv("returns.csv", parse_dates=['return_date'])
promos    = pd.read_csv("promotions.csv", parse_dates=['start_date','end_date'])

# feature engineering
df = df[df['order_status'] != 'cancelled'].copy()
df['revenue']       = df['unit_price'] * df['quantity'] - df['discount_amount']
df['margin']        = (df['price'] - df['cogs']) / df['price']
df['gross_profit']  = df['margin'] * df['revenue']
df['discount_rate'] = df['discount_amount'] / (df['unit_price'] * df['quantity'])
df['discount_rate'] = df['discount_rate'].fillna(0)
df['year']          = df['order_date'].dt.year
df['year_month']    = df['order_date'].dt.to_period('M')
df['has_promo']     = df['promo_id'].notna()

orders['year'] = orders['order_date'].dt.year

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1 — REVENUE DECLINE (2017 onwards)
# ══════════════════════════════════════════════════════════════════════════════
section("DIAGNOSTIC 1 · Revenue Decline")

# 1a. Decompose: volume vs AOV
yearly = df.groupby('year').agg(
    revenue    = ('revenue',  'sum'),
    orders_n   = ('order_id', 'nunique'),
    customers  = ('customer_id', 'nunique'),
).reset_index()
yearly['aov']            = yearly['revenue'] / yearly['orders_n']
yearly['orders_per_cust']= yearly['orders_n'] / yearly['customers']

# 1b. New vs returning customers
first_order = df.groupby('customer_id')['order_date'].min().dt.year.rename('first_year')
df2 = df.join(first_order, on='customer_id')
df2['cust_type'] = np.where(df2['year'] == df2['first_year'], 'New', 'Returning')
cust_type_rev = df2.groupby(['year','cust_type'])['revenue'].sum().unstack().fillna(0)

# 1c. Cancellation rate per year
cancel_rate = orders.groupby('year').apply(
    lambda x: (x['order_status']=='cancelled').mean()
).reset_index()
cancel_rate.columns = ['year', 'cancel_rate']

# 1d. Return rate per year
orders_per_year = orders.groupby('year')['order_id'].count().rename('total_orders')
returns['year'] = returns['return_date'].dt.year
ret_per_year    = returns.groupby('year')['order_id'].count().rename('returns_n')
return_rate     = pd.concat([orders_per_year, ret_per_year], axis=1).fillna(0)
return_rate['rate'] = return_rate['returns_n'] / return_rate['total_orders']

# 1e. Discount rate per year
disc_yearly = df.groupby('year')['discount_rate'].mean()

# ── PLOT D1 ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(BG)
fig.suptitle('DIAGNOSTIC 1  ·  WHY DID REVENUE DECLINE AFTER 2016?',
             fontsize=18, fontweight='bold', color=TEXT_HI, y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: Revenue + Orders + AOV
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(BG_MID)
color_rev = C_BLUE
ax1.fill_between(yearly['year'], yearly['revenue'], alpha=0.15, color=color_rev)
ax1.plot(yearly['year'], yearly['revenue'], color=color_rev, lw=2.5, marker='o', ms=6, label='Revenue')
ax1.set_ylabel('Revenue', color=color_rev)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_b))
ax1.tick_params(axis='y', labelcolor=color_rev)
ax1b = ax1.twinx()
ax1b.plot(yearly['year'], yearly['aov'], color=C_AMBER, lw=2, marker='s', ms=5,
          linestyle='--', label='AOV')
ax1b.plot(yearly['year'], yearly['orders_n']/1000, color=C_GREEN, lw=2, marker='^', ms=5,
          linestyle=':', label='Orders (K)')
ax1b.set_ylabel('AOV / Orders(K)', color=TEXT_LO)
ax1b.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_b))
ax1.set_title('A · Revenue vs. AOV vs. Order Volume', fontsize=12, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper right', framealpha=0.2)
ax1.axvline(2016, color=C_RED, lw=1, linestyle='--', alpha=0.6)
ax1.text(2016.1, yearly['revenue'].max()*0.92, '2016 Peak', color=C_RED, fontsize=8)
ax1.set_facecolor(BG_MID)

# Panel B: New vs Returning Revenue
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(BG_MID)
years_idx = cust_type_rev.index
if 'New' in cust_type_rev.columns:
    ax2.bar(years_idx, cust_type_rev['New'], color=C_GREEN, alpha=0.85, label='New', width=0.6)
if 'Returning' in cust_type_rev.columns:
    bottom = cust_type_rev['New'] if 'New' in cust_type_rev.columns else 0
    ax2.bar(years_idx, cust_type_rev['Returning'], bottom=bottom,
            color=C_BLUE, alpha=0.85, label='Returning', width=0.6)
ax2.set_title('B · New vs. Returning Customer Revenue', fontsize=11, fontweight='bold')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_b))
ax2.legend(fontsize=8, framealpha=0.2)

# Panel C: Cancellation rate
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(BG_MID)
cancel_data = orders.groupby('year').apply(
    lambda x: (x['order_status']=='cancelled').mean()
).reset_index()
cancel_data.columns = ['year','cancel_rate']
ax3.bar(cancel_data['year'], cancel_data['cancel_rate']*100,
        color=C_RED, alpha=0.8, width=0.6)
ax3.set_title('C · Cancellation Rate by Year (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Cancel Rate (%)')
for _, row in cancel_data.iterrows():
    ax3.text(row['year'], row['cancel_rate']*100 + 0.2,
             f"{row['cancel_rate']:.1%}", ha='center', fontsize=8, color=TEXT_HI)

# Panel D: Return rate
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor(BG_MID)
rr = return_rate.reset_index()
ax4.bar(rr['year'], rr['rate']*100, color=C_PURPLE, alpha=0.8, width=0.6)
ax4.set_title('D · Return Rate by Year (%)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Return Rate (%)')
for _, row in rr.iterrows():
    ax4.text(row['year'], row['rate']*100 + 0.05,
             f"{row['rate']:.1%}", ha='center', fontsize=8, color=TEXT_HI)

# Panel E: Discount rate
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor(BG_MID)
disc = disc_yearly.reset_index()
disc.columns = ['year','disc_rate']
ax5.plot(disc['year'], disc['disc_rate']*100, color=C_AMBER, lw=2.5, marker='o', ms=6)
ax5.fill_between(disc['year'], disc['disc_rate']*100, alpha=0.15, color=C_AMBER)
ax5.set_title('E · Avg Discount Rate by Year (%)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Discount Rate (%)')
ax5.axvline(2016, color=C_RED, lw=1, linestyle='--', alpha=0.6)

plt.savefig('diag1_revenue_decline.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("✅ diag1_revenue_decline.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2 — MARGIN LEAKAGE (Balanced + Premium)
# ══════════════════════════════════════════════════════════════════════════════
section("DIAGNOSTIC 2 · Margin Leakage")

seg_focus = ['Balanced', 'Everyday', 'Premium', 'Activewear', 'Performance']

# 2a. Discount rate by segment
disc_seg = df.groupby('segment')['discount_rate'].mean().sort_values(ascending=False)

# 2b. Promo dependency by segment
promo_dep = df.groupby('segment')['has_promo'].mean().sort_values(ascending=False)

# 2c. unit_price vs list price gap
df['price_gap_pct'] = (df['price'] - df['unit_price']) / df['price']
price_gap_seg = df.groupby('segment')['price_gap_pct'].mean().sort_values(ascending=False)

# 2d. Margin trend over time for Balanced vs Everyday
margin_trend = df[df['segment'].isin(['Balanced','Everyday','Premium','Activewear'])]\
    .groupby(['year','segment'])['margin'].mean().unstack()

# 2e. COGS vs Price by segment (box-like: mean ± std)
cogs_seg = df.groupby('segment').agg(
    avg_price = ('price', 'mean'),
    avg_cogs  = ('cogs',  'mean'),
    avg_margin= ('margin','mean'),
).reset_index().sort_values('avg_margin')

# 2f. Stackable promo orders
df['is_stackable'] = df['has_promo']  # proxy: has promo_id

# ── PLOT D2 ────────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(20, 16))
fig2.patch.set_facecolor(BG)
fig2.suptitle('DIAGNOSTIC 2  ·  WHY DO BALANCED & PREMIUM HAVE LOW MARGIN?',
              fontsize=18, fontweight='bold', color=TEXT_HI, y=0.98)

gs2 = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.45, wspace=0.35)

# Panel A: Discount rate by segment
ax = fig2.add_subplot(gs2[0, 0])
ax.set_facecolor(BG_MID)
colors_bar = [C_RED if v > disc_seg.median() else C_GREEN for v in disc_seg.values]
bars = ax.barh(disc_seg.index, disc_seg.values*100, color=colors_bar, alpha=0.85, height=0.6)
for bar, val in zip(bars, disc_seg.values):
    ax.text(val*100+0.1, bar.get_y()+bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=8, color=TEXT_HI)
ax.axvline(disc_seg.median()*100, color=TEXT_LO, lw=1, linestyle='--', alpha=0.6)
ax.set_title('A · Avg Discount Rate by Segment', fontsize=11, fontweight='bold')
ax.set_xlabel('Discount Rate (%)')

# Panel B: Promo dependency
ax = fig2.add_subplot(gs2[0, 1])
ax.set_facecolor(BG_MID)
colors_bar2 = [C_RED if v > promo_dep.median() else C_GREEN for v in promo_dep.values]
bars = ax.barh(promo_dep.index, promo_dep.values*100, color=colors_bar2, alpha=0.85, height=0.6)
for bar, val in zip(bars, promo_dep.values):
    ax.text(val*100+0.3, bar.get_y()+bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=8, color=TEXT_HI)
ax.axvline(promo_dep.median()*100, color=TEXT_LO, lw=1, linestyle='--', alpha=0.6)
ax.set_title('B · Promo Dependency by Segment (% orders with promo)', fontsize=11, fontweight='bold')
ax.set_xlabel('% Orders Using Promo')

# Panel C: unit_price vs list price gap
ax = fig2.add_subplot(gs2[1, 0])
ax.set_facecolor(BG_MID)
colors_gap = [C_RED if v > price_gap_seg.median() else C_AMBER for v in price_gap_seg.values]
bars = ax.barh(price_gap_seg.index, price_gap_seg.values*100, color=colors_gap, alpha=0.85, height=0.6)
for bar, val in zip(bars, price_gap_seg.values):
    ax.text(val*100+0.1, bar.get_y()+bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=8, color=TEXT_HI)
ax.set_title('C · Price Gap: List Price vs. Actual Unit Price (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Gap % (higher = selling below list price more)')

# Panel D: Margin trend over time
ax = fig2.add_subplot(gs2[1, 1])
ax.set_facecolor(BG_MID)
seg_colors = {'Balanced': C_RED, 'Everyday': C_GREEN, 'Premium': C_AMBER, 'Activewear': C_BLUE}
for seg in margin_trend.columns:
    col = seg_colors.get(seg, TEXT_LO)
    ax.plot(margin_trend.index, margin_trend[seg]*100, color=col, lw=2,
            marker='o', ms=4, label=seg)
ax.set_title('D · Margin Trend Over Time by Segment', fontsize=11, fontweight='bold')
ax.set_ylabel('Gross Margin (%)')
ax.legend(fontsize=8, framealpha=0.2)

# Panel E: COGS vs Price waterfall
ax = fig2.add_subplot(gs2[2, :])
ax.set_facecolor(BG_MID)
x = np.arange(len(cogs_seg))
w = 0.35
bars1 = ax.bar(x - w/2, cogs_seg['avg_price'], width=w, color=C_BLUE, alpha=0.85, label='Avg List Price')
bars2 = ax.bar(x + w/2, cogs_seg['avg_cogs'],  width=w, color=C_RED,  alpha=0.85, label='Avg COGS')
ax.set_xticks(x)
ax.set_xticklabels(cogs_seg['segment'], rotation=0)
ax.set_title('E · Avg List Price vs. COGS by Segment — margin labeled', fontsize=11, fontweight='bold')
ax.set_ylabel('VND')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_b))
ax.legend(fontsize=9, framealpha=0.2)
for i, row in enumerate(cogs_seg.itertuples()):
    ax.text(i, row.avg_price + 2000, f'M:{row.avg_margin:.1%}',
            ha='center', fontsize=8, color=C_AMBER, fontweight='bold')

plt.savefig('diag2_margin_leakage.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("✅ diag2_margin_leakage.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2 — PROMO EFFECTIVENESS
# ══════════════════════════════════════════════════════════════════════════════

# Style
plt.rcParams.update({
    'figure.facecolor': '#0B0E1A', 'axes.facecolor': '#13172A',
    'axes.edgecolor': '#252A45',   'axes.labelcolor': '#B8BDD6',
    'axes.titlecolor': '#FFFFFF',  'xtick.color': '#6B7099',
    'ytick.color': '#6B7099',      'text.color': '#B8BDD6',
    'grid.color': '#252A45',       'grid.linestyle': '--',
    'grid.alpha': 0.5,             'font.family': 'monospace',
    'axes.spines.top': False,      'axes.spines.right': False,
})
BG = '#0B0E1A'; BG_MID = '#13172A'
C_BLUE = '#4F8EF7'; C_RED = '#F75A5A'; C_GREEN = '#4FF7A0'
C_AMBER = '#F7C94F'; C_PURPLE = '#BF4FF7'
TEXT_HI = '#FFFFFF'; TEXT_LO = '#6B7099'

def fmt_b(x, _=None):
    if abs(x) >= 1e9: return f'{x/1e9:.2f}B'
    if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
    if abs(x) >= 1e3: return f'{x/1e3:.0f}K'
    return f'{x:.1f}'

# Load
df = pd.read_csv("df_1.csv", parse_dates=['order_date'])
df = df[df['order_status'] != 'cancelled'].copy()

# Feature engineering
df['gross_revenue'] = df['unit_price'] * df['quantity']   # BEFORE discount
df['net_revenue']   = df['gross_revenue'] - df['discount_amount']  # AFTER discount
df['margin']        = (df['price'] - df['cogs']) / df['price']
df['has_promo']     = df['promo_id'].notna().astype(int)  # 1 = promo, 0 = no promo
df['year']          = df['order_date'].dt.year
df['discount_rate'] = df['discount_amount'] / df['gross_revenue'].replace(0, np.nan)

# ── ORDER-LEVEL aggregation (correct unit for AOV) ────────────────────────────
order_agg = df.groupby(['order_id', 'has_promo', 'year']).agg(
    gross_order_val = ('gross_revenue',   'sum'),
    net_order_val   = ('net_revenue',     'sum'),
    discount_amt    = ('discount_amount', 'sum'),
    quantity        = ('quantity',        'sum'),
    margin          = ('margin',          'mean'),
).reset_index()

# 3a. Promo vs No-promo comparison (ORDER level) 
promo_compare = order_agg.groupby('has_promo').agg(
    gross_revenue = ('gross_order_val', 'sum'),
    net_revenue   = ('net_order_val',   'sum'),
    avg_gross_aov = ('gross_order_val', 'mean'),
    avg_net_aov   = ('net_order_val',   'mean'),
    avg_discount  = ('discount_amt',    'mean'),
    avg_quantity  = ('quantity',        'mean'),
    avg_margin    = ('margin',          'mean'),
    n_orders      = ('order_id',        'count'),
).reset_index()
promo_compare['label']        = promo_compare['has_promo'].map({1:'With Promo', 0:'No Promo'})
promo_compare['discount_pct'] = promo_compare['avg_discount'] / promo_compare['avg_gross_aov']

print("=== Promo vs No-Promo (Order Level) ===")
print(promo_compare[['label','avg_gross_aov','avg_net_aov',
                      'avg_discount','discount_pct','avg_quantity',
                      'avg_margin','n_orders']].to_string(index=False))

# 3b. Over time
disc_time  = df.groupby('year').agg(
    gross_revenue = ('gross_revenue', 'sum'),
    disc_rate     = ('discount_rate', 'mean'),
    margin        = ('margin',        'mean'),
).reset_index()
promo_time = df.groupby('year')['has_promo'].mean()

# ── 3c. Margin cost by category (gross revenue basis) ────────────────────────
cat_promo  = df.groupby(['category','has_promo']).agg(
    gross_revenue = ('gross_revenue', 'sum'),
    margin        = ('margin',        'mean'),
    orders        = ('order_id',      'nunique'),
).reset_index()

promo_lift = cat_promo.pivot_table(
    index='category', columns='has_promo',
    values=['gross_revenue','margin','orders']).fillna(0)
promo_lift.columns = ['_'.join([str(c) for c in col]) for col in promo_lift.columns]

if 'margin_1' in promo_lift.columns and 'margin_0' in promo_lift.columns:
    promo_lift['margin_cost'] = promo_lift['margin_0'] - promo_lift['margin_1']

# ── 3d. Gross AOV promo vs no-promo over time ─────────────────────────────────
aov_time = order_agg.groupby(['year','has_promo'])['gross_order_val'].mean().unstack().fillna(0)

# ── PLOT ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor(BG)
fig.suptitle('DIAGNOSTIC 3  ·  ARE PROMOTIONS BUYING REVENUE AT THE COST OF MARGIN?\n'
             '(Revenue = unit_price × quantity, BEFORE discount — apples-to-apples comparison)',
             fontsize=15, fontweight='bold', color=TEXT_HI, y=0.99)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

# ── Panel A: Gross AOV — promo vs no-promo ────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor(BG_MID)
labels   = promo_compare['label']
aov_vals = promo_compare['avg_gross_aov']
colors_p = [C_AMBER, C_BLUE]
bars = ax.bar(labels, aov_vals, color=colors_p, alpha=0.85, width=0.5)
for bar, val in zip(bars, aov_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val*1.015,
            fmt_b(val), ha='center', fontsize=11, fontweight='bold', color=TEXT_HI)

# Annotate qty difference
qty_promo  = promo_compare.loc[promo_compare['has_promo']==1, 'avg_quantity'].values[0]
qty_nopromo= promo_compare.loc[promo_compare['has_promo']==0, 'avg_quantity'].values[0]
ax.text(0.5, 0.08,
        f'Avg qty/order:  With Promo={qty_promo:.1f}  |  No Promo={qty_nopromo:.1f}',
        transform=ax.transAxes, ha='center', fontsize=8, color=TEXT_LO,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, alpha=0.7))

ax.set_title('A · Gross AOV (before discount): Promo vs. No Promo\n'
             'Apples-to-apples — discount effect removed',
             fontsize=10, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_b))
ax.set_ylabel('Gross AOV (VND)')

# ── Panel B: Margin + Discount rate — promo vs no-promo ───────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor(BG_MID)
x      = np.arange(len(labels))
w_bar  = 0.35
margin_vals  = promo_compare['avg_margin'] * 100
disc_vals    = promo_compare['discount_pct'] * 100

b1 = ax.bar(x - w_bar/2, margin_vals, width=w_bar,
            color=colors_p, alpha=0.85, label='Avg Margin %')
b2 = ax.bar(x + w_bar/2, disc_vals,   width=w_bar,
            color=[C_RED, C_GREEN], alpha=0.6, label='Avg Discount %')

for bar, val in zip(b1, margin_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.2,
            f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color=TEXT_HI)
for bar, val in zip(b2, disc_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.2,
            f'{val:.1f}%', ha='center', fontsize=9, color=TEXT_HI)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('B · Avg Margin % vs. Avg Discount % by Promo Status',
             fontsize=10, fontweight='bold')
ax.set_ylabel('%')
ax.legend(fontsize=8, framealpha=0.2)

# ── Panel C: Revenue + Promo usage + Margin over time ────────────────────────
ax = fig.add_subplot(gs[1, :])
ax.set_facecolor(BG_MID)
ax.fill_between(disc_time['year'], disc_time['gross_revenue'], alpha=0.12, color=C_BLUE)
ax.plot(disc_time['year'], disc_time['gross_revenue'],
        color=C_BLUE, lw=2.5, marker='o', ms=5, label='Gross Revenue')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_b))
ax.set_ylabel('Gross Revenue', color=C_BLUE)
ax.tick_params(axis='y', labelcolor=C_BLUE)

ax2 = ax.twinx()
ax2.plot(promo_time.index, promo_time.values*100,
         color=C_AMBER, lw=2.5, marker='s', ms=5,
         linestyle='--', label='% Orders w/ Promo')
ax2.plot(disc_time['year'], disc_time['margin']*100,
         color=C_RED, lw=2, marker='^', ms=5,
         linestyle=':', label='Avg Margin %')
ax2.set_ylabel('Promo Usage % / Margin %', color=TEXT_LO)

ax.set_title('C · Gross Revenue vs. Promo Usage Rate vs. Margin Over Time',
             fontsize=11, fontweight='bold')
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbl1+lbl2, fontsize=8, loc='upper right', framealpha=0.2)

# ── Panel D: Margin cost of promo by category ─────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
ax.set_facecolor(BG_MID)
if 'margin_cost' in promo_lift.columns:
    mc = promo_lift['margin_cost'].sort_values(ascending=False)
    colors_mc = [C_RED if v > 0 else C_GREEN for v in mc.values]
    bars = ax.barh(mc.index, mc.values*100, color=colors_mc, alpha=0.85, height=0.55)
    for bar, val in zip(bars, mc.values):
        offset = 0.05 if val >= 0 else -0.3
        ax.text(val*100 + offset, bar.get_y()+bar.get_height()/2,
                f'{val:.2%}', va='center', fontsize=8, color=TEXT_HI)
    ax.axvline(0, color=TEXT_LO, lw=1)
    ax.set_title('D · Margin Difference by Category\n'
                 'No Promo − With Promo  (positive = promo hurts margin)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Margin Difference (%)')

# ── Panel E: Gross AOV promo vs no-promo over time ───────────────────────────
ax = fig.add_subplot(gs[2, 1])
ax.set_facecolor(BG_MID)
if 1 in aov_time.columns:
    ax.plot(aov_time.index, aov_time[1],
            color=C_AMBER, lw=2.5, marker='o', ms=5, label='With Promo')
if 0 in aov_time.columns:
    ax.plot(aov_time.index, aov_time[0],
            color=C_BLUE, lw=2.5, marker='s', ms=5,
            linestyle='--', label='No Promo')

# Annotate gap
if 1 in aov_time.columns and 0 in aov_time.columns:
    last_yr   = aov_time.index[-1]
    gap_pct   = (aov_time.loc[last_yr, 1] - aov_time.loc[last_yr, 0]) / \
                 aov_time.loc[last_yr, 0] * 100
    ax.text(0.03, 0.08, f'Gap (latest year): {gap_pct:+.1f}%',
            transform=ax.transAxes, fontsize=8, color=TEXT_LO,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, alpha=0.7))

ax.set_title('E · Gross AOV Over Time: Promo vs. No Promo\n'
             '(unit_price × qty — discount NOT subtracted)',
             fontsize=10, fontweight='bold')
ax.set_ylabel('Gross AOV (VND)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_b))
ax.legend(fontsize=9, framealpha=0.2)

plt.savefig('diag3_promo_fixed.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("✅ diag3_promo_fixed.png saved")

# ── Summary print ──────────────────────────────────────────────────────────────
p  = promo_compare[promo_compare['has_promo']==1].iloc[0]
np_ = promo_compare[promo_compare['has_promo']==0].iloc[0]
aov_gap = (p['avg_gross_aov'] - np_['avg_gross_aov']) / np_['avg_gross_aov'] * 100
qty_gap = (p['avg_quantity']  - np_['avg_quantity'])  / np_['avg_quantity']  * 100

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINT
# ══════════════════════════════════════════════════════════════════════════════
section("DIAGNOSTIC SUMMARY")

# D1
new_rev_2012 = cust_type_rev.loc[2012,'New'] if 2012 in cust_type_rev.index and 'New' in cust_type_rev.columns else 0
cancel_peak  = cancel_data.loc[cancel_data['cancel_rate'].idxmax()]
disc_2022    = disc_time[disc_time['year']==2022]['disc_rate'].values[0] if 2022 in disc_time['year'].values else 0
disc_2012    = disc_time[disc_time['year']==2012]['disc_rate'].values[0] if 2012 in disc_time['year'].values else 0

print(f"""
DIAGNOSTIC 1 — Revenue Decline
  Cancel rate peak year : {int(cancel_peak['year'])} ({cancel_peak['cancel_rate']:.1%})
  Discount rate 2012    : {disc_2012:.1%}
  Discount rate 2022    : {disc_2022:.1%}

DIAGNOSTIC 2 — Margin Leakage
  Segment w/ highest discount rate : {disc_seg.idxmax()} ({disc_seg.max():.1%})
  Segment w/ highest promo depend. : {promo_dep.idxmax()} ({promo_dep.max():.1%})

DIAGNOSTIC 3 — Promo Effectiveness
                      With Promo    No Promo
Gross AOV (before disc) {fmt_b(p['avg_gross_aov']):>10}  {fmt_b(np_['avg_gross_aov']):>10}
AOV Gap                 {aov_gap:>+9.1f}%
Avg Discount            {fmt_b(p['avg_discount']):>10}  {fmt_b(np_['avg_discount']):>10}
Avg Discount %          {p['discount_pct']:>9.1%}  {np_['discount_pct']:>9.1%}
Avg Margin              {p['avg_margin']:>9.1%}  {np_['avg_margin']:>9.1%}
Avg Qty / Order         {p['avg_quantity']:>10.2f}  {np_['avg_quantity']:>10.2f}
Qty Gap                 {qty_gap:>+9.1f}%
N Orders                {int(p['n_orders']):>10,}  {int(np_['n_orders']):>10,}
""")

# Predictive Analysis
# Style
plt.rcParams.update({
    'figure.facecolor': '#0B0E1A', 'axes.facecolor': '#13172A',
    'axes.edgecolor': '#252A45',   'axes.labelcolor': '#B8BDD6',
    'axes.titlecolor': '#FFFFFF',  'xtick.color': '#6B7099',
    'ytick.color': '#6B7099',      'text.color': '#B8BDD6',
    'grid.color': '#252A45',       'grid.linestyle': '--',
    'grid.alpha': 0.5,             'font.family': 'monospace',
    'axes.spines.top': False,      'axes.spines.right': False,
})
BG = '#0B0E1A'; BG_MID = '#13172A'
C_BLUE = '#4F8EF7'; C_RED = '#F75A5A'; C_GREEN = '#4FF7A0'; C_AMBER = '#F7C94F'
TEXT_HI = '#FFFFFF'; TEXT_LO = '#6B7099'
 
def fmt_b(x, _=None):
    if abs(x) >= 1e9: return f'{x/1e9:.2f}B'
    if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
    if abs(x) >= 1e3: return f'{x/1e3:.0f}K'
    return f'{x:.1f}'
 
# 1. Load & prep
sales  = pd.read_csv("sales.csv",       parse_dates=['Date'])
web    = pd.read_csv("web_traffic.csv", parse_dates=['date'])
promos = pd.read_csv("promotions.csv",  parse_dates=['start_date','end_date'])
 
sales   = sales.sort_values('Date').set_index('Date')
monthly = sales['Revenue'].resample('MS').sum().reset_index()
monthly.columns = ['ds', 'y']
 
# 2. Feature engineering 
web_monthly = web.groupby(pd.Grouper(key='date', freq='MS')).agg(
    sessions        = ('sessions',                'sum'),
    unique_visitors = ('unique_visitors',          'sum'),
    bounce_rate     = ('bounce_rate',              'mean'),
    avg_duration    = ('avg_session_duration_sec', 'mean'),
).reset_index().rename(columns={'date': 'ds'})
 
promo_counts = []
for d in pd.date_range(monthly['ds'].min(), monthly['ds'].max(), freq='MS'):
    eom    = d + pd.offsets.MonthEnd(1)
    active = ((promos['start_date'] <= eom) & (promos['end_date'] >= d)).sum()
    promo_counts.append({'ds': d, 'active_promos': active})
promo_monthly = pd.DataFrame(promo_counts)
 
feat = monthly.merge(web_monthly,   on='ds', how='left') \
              .merge(promo_monthly, on='ds', how='left')
feat = feat.ffill().fillna(0)
 
feat['month']     = feat['ds'].dt.month
feat['year']      = feat['ds'].dt.year
feat['quarter']   = feat['ds'].dt.quarter
feat['month_sin'] = np.sin(2 * np.pi * feat['month'] / 12)
feat['month_cos'] = np.cos(2 * np.pi * feat['month'] / 12)
feat['trend']     = np.arange(len(feat))
 
for lag in [1, 2, 3, 6, 12]:
    feat[f'lag_{lag}'] = feat['y'].shift(lag)
for window in [3, 6, 12]:
    feat[f'roll_mean_{window}'] = feat['y'].shift(1).rolling(window).mean()
    feat[f'roll_std_{window}']  = feat['y'].shift(1).rolling(window).std()
 
feat = feat.dropna().reset_index(drop=True)
 
FEATURES = ['trend','month_sin','month_cos','quarter',
            'sessions','unique_visitors','bounce_rate','avg_duration',
            'active_promos',
            'lag_1','lag_2','lag_3','lag_6','lag_12',
            'roll_mean_3','roll_mean_6','roll_mean_12',
            'roll_std_3','roll_std_6','roll_std_12']
 
# 3. Split 
train = feat[feat['year'] <= 2020]
val   = feat[feat['year'] == 2021]
test  = feat[feat['year'] == 2022]
 
X_tr, y_tr = train[FEATURES], train['y']
X_va, y_va = val[FEATURES],   val['y']
X_te, y_te = test[FEATURES],  test['y']
 
# 4. Models 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
 
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)
 
ridge = Ridge(alpha=10)
ridge.fit(X_tr_s, y_tr)
 
rf = RandomForestRegressor(n_estimators=300, max_depth=6,
                            min_samples_leaf=3, random_state=42)
rf.fit(X_tr, y_tr)
 
gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                max_depth=4, min_samples_leaf=3,
                                subsample=0.8, random_state=42)
gb.fit(X_tr, y_tr)
 
mr = mean_absolute_percentage_error(y_va, ridge.predict(X_va_s))
mf = mean_absolute_percentage_error(y_va, rf.predict(X_va))
mg = mean_absolute_percentage_error(y_va, gb.predict(X_va))
wts = np.array([1/mr, 1/mf, 1/mg]); wts = wts / wts.sum()
 
def ens(Xs, X):
    return wts[0]*ridge.predict(Xs) + wts[1]*rf.predict(X) + wts[2]*gb.predict(X)
 
val_pred  = ens(X_va_s, X_va)
test_pred = ens(X_te_s, X_te)
mape_val  = mean_absolute_percentage_error(y_va, val_pred)
mape_test = mean_absolute_percentage_error(y_te, test_pred)
 
print(f"Val MAPE : {mape_val:.2%}")
print(f"Test MAPE: {mape_test:.2%}")
 
# 5. Forecast 18 months (Jan 2023 – Jun 2024) 
history_y   = list(feat['y'].values)
future_rows = []
 
for i in range(18):
    next_ds    = feat['ds'].max() + pd.DateOffset(months=i+1)
    next_month = next_ds.month
    next_trend = feat['trend'].max() + i + 1
 
    last_sess = web_monthly.iloc[-1]["sessions"]
    last_vis  = web_monthly.iloc[-1]["unique_visitors"]
    past_sess = [r["sessions"] for r in future_rows if "sessions" in r][-3:]
    past_vis  = [r["unique_visitors"] for r in future_rows if "unique_visitors" in r][-3:]
    avg_sess  = np.mean(past_sess + [last_sess])
    avg_vis   = np.mean(past_vis  + [last_vis])
 
    lags  = {f'lag_{l}': history_y[len(history_y)-l] for l in [1,2,3,6,12]}
    rolls = {f'roll_mean_{w}': np.mean(history_y[-w:]) for w in [3,6,12]}
    stds  = {f'roll_std_{w}':  np.std(history_y[-w:])  for w in [3,6,12]}
 
    row = pd.DataFrame([{
        'ds': next_ds, 'trend': next_trend,
        'month_sin':       np.sin(2*np.pi*next_month/12),
        'month_cos':       np.cos(2*np.pi*next_month/12),
        'quarter':         (next_month-1)//3 + 1,
        'sessions':        avg_sess,
        'unique_visitors': avg_vis,
        'bounce_rate':     web_monthly['bounce_rate'].mean(),
        'avg_duration':    web_monthly['avg_duration'].mean(),
        'active_promos':   promo_monthly['active_promos'].mean(),
        **lags, **rolls, **stds
    }])[FEATURES]
 
    pred = max(0, ens(scaler.transform(row), row)[0])
    history_y.append(pred)
    future_rows.append({'ds': next_ds, 'y_pred': pred})
 
future_df = pd.DataFrame(future_rows)
 
# 6. Single plot 
fig, ax = plt.subplots(figsize=(22, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG_MID)
 
ax.fill_between(feat['ds'], feat['y'], alpha=0.10, color=C_BLUE)
ax.plot(feat['ds'], feat['y'],
        color=C_BLUE, lw=1.8, label='Actual Revenue (Monthly)')
ax.plot(val['ds'],  val_pred,
        color=C_AMBER, lw=2, linestyle='--',
        label=f'Predicted — Val 2021  (MAPE {mape_val:.1%})')
ax.plot(test['ds'], test_pred,
        color=C_RED, lw=2, linestyle='--',
        label=f'Predicted — Test 2022  (MAPE {mape_test:.1%})')
ax.plot(future_df['ds'], future_df['y_pred'],
        color=C_GREEN, lw=2.5, marker='o', ms=5,
        label='Forecast Jan 2023 – Jun 2024')
ax.fill_between(future_df['ds'], future_df['y_pred'],
                alpha=0.12, color=C_GREEN)
ax.axvspan(future_df['ds'].min(), future_df['ds'].max(),
           alpha=0.04, color=C_GREEN)
ax.axvline(pd.Timestamp('2023-01-01'),
           color=C_GREEN, lw=1.5, linestyle=':', alpha=0.8)
 
# Peak annotation
peak_idx  = feat['y'].idxmax()
peak_date = feat.loc[peak_idx, 'ds']
peak_val  = feat.loc[peak_idx, 'y']
ax.annotate('2016 Peak', xy=(peak_date, peak_val),
            xytext=(peak_date - pd.DateOffset(months=22), peak_val * 0.86),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
            color=C_RED, fontsize=8)
ax.text(pd.Timestamp('2023-02-01'),
        future_df['y_pred'].max() * 1.06,
        'Forecast →', color=C_GREEN, fontsize=8, alpha=0.8)
 
# Summary box
avg_2016     = feat[feat['year'] == 2016]['y'].mean()
avg_forecast = future_df['y_pred'].mean()
decline_pct  = (avg_2016 - avg_forecast) / avg_2016 * 100
 
box = (
    f"Model     Ridge + RF + GBM (Ensemble)\n"
    f"Weights   {wts[0]:.2f} / {wts[1]:.2f} / {wts[2]:.2f}\n"
    f"────────────────────────────────\n"
    f"Val  MAPE   {mape_val:.1%}\n"
    f"Test MAPE   {mape_test:.1%}\n"
    f"────────────────────────────────\n"
    f"Avg/month 2016 peak   {fmt_b(avg_2016)}\n"
    f"Avg/month 2023–24     {fmt_b(avg_forecast)}\n"
    f"Decline from peak     {decline_pct:.0f}%"
)
ax.text(0.995, 0.97, box, transform=ax.transAxes,
        fontsize=8, va='top', ha='right', family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor=BG_MID,
                  edgecolor='#252A45', alpha=0.95),
        color=TEXT_HI)
 
ax.set_title(
    'TRAJECTORY 1  ·  REVENUE FORECAST 2023–2024\n'
    'Trend giảm dài hạn chưa có dấu hiệu đảo chiều nếu không có can thiệp',
    fontsize=14, fontweight='bold', color=TEXT_HI, pad=14)
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Revenue (VND)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_b))
ax.legend(fontsize=9, framealpha=0.2, loc='upper left')
 
plt.tight_layout()
plt.savefig('trajectory1_revenue_forecast.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("✅ trajectory1_revenue_forecast.png saved")
 
print("\nForecast (2023–2024):")
print(future_df.assign(
    Month   = future_df['ds'].dt.strftime('%Y-%m'),
    Revenue = future_df['y_pred'].apply(fmt_b)
)[['Month','Revenue']].to_string(index=False))

# Prescriptive analysis (predictive model alteration after solution)

# Style
plt.rcParams.update({
    'figure.facecolor': '#0B0E1A', 'axes.facecolor': '#13172A',
    'axes.edgecolor': '#252A45',   'axes.labelcolor': '#B8BDD6',
    'axes.titlecolor': '#FFFFFF',  'xtick.color': '#6B7099',
    'ytick.color': '#6B7099',      'text.color': '#B8BDD6',
    'grid.color': '#252A45',       'grid.linestyle': '--',
    'grid.alpha': 0.5,             'font.family': 'monospace',
    'axes.spines.top': False,      'axes.spines.right': False,
})
BG = '#0B0E1A'; BG_MID = '#13172A'
C_BLUE   = '#4F8EF7'; C_RED    = '#F75A5A'
C_GREEN  = '#4FF7A0'; C_AMBER  = '#F7C94F'
C_PURPLE = '#BF4FF7'; C_TEAL   = '#4FF7F0'
TEXT_HI  = '#FFFFFF'; TEXT_LO  = '#6B7099'

def fmt_b(x, _=None):
    if abs(x) >= 1e9: return f'{x/1e9:.2f}B'
    if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
    if abs(x) >= 1e3: return f'{x/1e3:.0f}K'
    return f'{x:.1f}'

# ══════════════════════════════════════════════════════════════════════════════
# 1. REBUILD MODEL (same as trajectory1)
# ══════════════════════════════════════════════════════════════════════════════
sales  = pd.read_csv("sales.csv",       parse_dates=['Date'])
web    = pd.read_csv("web_traffic.csv", parse_dates=['date'])
promos = pd.read_csv("promotions.csv",  parse_dates=['start_date','end_date'])

sales   = sales.sort_values('Date').set_index('Date')
monthly = sales['Revenue'].resample('MS').sum().reset_index()
monthly.columns = ['ds', 'y']

web_monthly = web.groupby(pd.Grouper(key='date', freq='MS')).agg(
    sessions        = ('sessions',                'sum'),
    unique_visitors = ('unique_visitors',          'sum'),
    bounce_rate     = ('bounce_rate',              'mean'),
    avg_duration    = ('avg_session_duration_sec', 'mean'),
).reset_index().rename(columns={'date': 'ds'})

promo_counts = []
for d in pd.date_range(monthly['ds'].min(), monthly['ds'].max(), freq='MS'):
    eom    = d + pd.offsets.MonthEnd(1)
    active = ((promos['start_date'] <= eom) & (promos['end_date'] >= d)).sum()
    promo_counts.append({'ds': d, 'active_promos': active})
promo_monthly = pd.DataFrame(promo_counts)

feat = monthly.merge(web_monthly,   on='ds', how='left') \
              .merge(promo_monthly, on='ds', how='left')
feat = feat.ffill().fillna(0)

feat['month']     = feat['ds'].dt.month
feat['year']      = feat['ds'].dt.year
feat['quarter']   = feat['ds'].dt.quarter
feat['month_sin'] = np.sin(2 * np.pi * feat['month'] / 12)
feat['month_cos'] = np.cos(2 * np.pi * feat['month'] / 12)
feat['trend']     = np.arange(len(feat))

for lag in [1, 2, 3, 6, 12]:
    feat[f'lag_{lag}'] = feat['y'].shift(lag)
for window in [3, 6, 12]:
    feat[f'roll_mean_{window}'] = feat['y'].shift(1).rolling(window).mean()
    feat[f'roll_std_{window}']  = feat['y'].shift(1).rolling(window).std()

feat = feat.dropna().reset_index(drop=True)

FEATURES = ['trend','month_sin','month_cos','quarter',
            'sessions','unique_visitors','bounce_rate','avg_duration',
            'active_promos',
            'lag_1','lag_2','lag_3','lag_6','lag_12',
            'roll_mean_3','roll_mean_6','roll_mean_12',
            'roll_std_3','roll_std_6','roll_std_12']

train = feat[feat['year'] <= 2020]
val   = feat[feat['year'] == 2021]
test  = feat[feat['year'] == 2022]

X_tr, y_tr = train[FEATURES], train['y']
X_va, y_va = val[FEATURES],   val['y']
X_te, y_te = test[FEATURES],  test['y']

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

ridge = Ridge(alpha=10); ridge.fit(X_tr_s, y_tr)
rf    = RandomForestRegressor(n_estimators=300, max_depth=6,
                               min_samples_leaf=3, random_state=42)
rf.fit(X_tr, y_tr)
gb    = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=4, min_samples_leaf=3,
                                   subsample=0.8, random_state=42)
gb.fit(X_tr, y_tr)

mr = mean_absolute_percentage_error(y_va, ridge.predict(X_va_s))
mf = mean_absolute_percentage_error(y_va, rf.predict(X_va))
mg = mean_absolute_percentage_error(y_va, gb.predict(X_va))
wts = np.array([1/mr, 1/mf, 1/mg]); wts = wts / wts.sum()

def ens(Xs, X):
    return wts[0]*ridge.predict(Xs) + wts[1]*rf.predict(X) + wts[2]*gb.predict(X)

val_pred  = ens(X_va_s, X_va)
test_pred = ens(X_te_s, X_te)
mape_val  = mean_absolute_percentage_error(y_va, val_pred)
mape_test = mean_absolute_percentage_error(y_te, test_pred)

print(f"Model ready | Val MAPE: {mape_val:.2%} | Test MAPE: {mape_test:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SCENARIO DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
# Base values from last 6 months of historical data
base_sess   = web_monthly.tail(6)['sessions'].mean()
base_vis    = web_monthly.tail(6)['unique_visitors'].mean()
base_bounce = web_monthly['bounce_rate'].mean()
base_dur    = web_monthly['avg_duration'].mean()
base_promos = promo_monthly['active_promos'].mean()

SCENARIOS = {
    'Baseline\n(No Intervention)': {
        'sess_mult':  1.00, 'vis_mult':  1.00, 'promo_add': 0,
        'color': TEXT_LO,  'linestyle': '--',
        'label': 'Baseline (No Intervention)',
    },
    'Scenario 1\nModerate Recovery': {
        'sess_mult':  1.20, 'vis_mult':  1.15, 'promo_add': 1,
        'color': C_AMBER,  'linestyle': '-',
        'label': 'S1: Moderate (+20% sessions, +1 promo/mo)',
        'action': '+20% sessions | +15% visitors | +1 promo/month',
    },
    'Scenario 2\nAggressive Growth': {
        'sess_mult':  1.40, 'vis_mult':  1.30, 'promo_add': 2,
        'color': C_GREEN,  'linestyle': '-',
        'label': 'S2: Aggressive (+40% sessions, +2 promos/mo)',
        'action': '+40% sessions | +30% visitors | +2 promos/month',
    },
    'Scenario 3\nOrganic SEO Only': {
        'sess_mult':  1.30, 'vis_mult':  1.25, 'promo_add': 0,
        'color': C_TEAL,   'linestyle': '-',
        'label': 'S3: Organic SEO (+30% sessions, no new promos)',
        'action': '+30% sessions | +25% visitors | promos unchanged',
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. FORECAST FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def run_forecast(sess_mult, vis_mult, promo_add, n_months=18):
    history_y   = list(feat['y'].values)
    future_rows = []

    for i in range(n_months):
        next_ds    = feat['ds'].max() + pd.DateOffset(months=i+1)
        next_month = next_ds.month
        next_trend = feat['trend'].max() + i + 1

        # Apply scenario multipliers
        adj_sess  = base_sess  * sess_mult
        adj_vis   = base_vis   * vis_mult
        adj_promos = base_promos + promo_add

        lags  = {f'lag_{l}': history_y[len(history_y)-l] for l in [1,2,3,6,12]}
        rolls = {f'roll_mean_{w}': np.mean(history_y[-w:]) for w in [3,6,12]}
        stds  = {f'roll_std_{w}':  np.std(history_y[-w:])  for w in [3,6,12]}

        row = pd.DataFrame([{
            'ds': next_ds, 'trend': next_trend,
            'month_sin':       np.sin(2*np.pi*next_month/12),
            'month_cos':       np.cos(2*np.pi*next_month/12),
            'quarter':         (next_month-1)//3 + 1,
            'sessions':        adj_sess,
            'unique_visitors': adj_vis,
            'bounce_rate':     base_bounce,
            'avg_duration':    base_dur,
            'active_promos':   adj_promos,
            **lags, **rolls, **stds
        }])[FEATURES]

        pred = max(0, ens(scaler.transform(row), row)[0])
        history_y.append(pred)
        future_rows.append({'ds': next_ds, 'y_pred': pred,
                            'sessions': adj_sess, 'unique_visitors': adj_vis})

    return pd.DataFrame(future_rows)

# Run all scenarios
results = {}
for name, cfg in SCENARIOS.items():
    results[name] = run_forecast(cfg['sess_mult'], cfg['vis_mult'], cfg['promo_add'])
    total = results[name]['y_pred'].sum()
    print(f"{name.replace(chr(10),' '):40s} Total 18M: {fmt_b(total)}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SINGLE PLOT
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(22, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG_MID)

# Historical
ax.fill_between(feat['ds'], feat['y'], alpha=0.08, color=C_BLUE)
ax.plot(feat['ds'], feat['y'],
        color=C_BLUE, lw=1.6, alpha=0.8, label='Actual Revenue (Monthly)')

# Val / Test fit
ax.plot(val['ds'],  val_pred,  color=C_BLUE, lw=1.5, linestyle=':', alpha=0.5)
ax.plot(test['ds'], test_pred, color=C_BLUE, lw=1.5, linestyle=':', alpha=0.5)

# Forecast zone
ax.axvline(pd.Timestamp('2023-01-01'),
           color=TEXT_LO, lw=1.2, linestyle=':', alpha=0.6)
ax.axvspan(results[list(results.keys())[0]]['ds'].min(),
           results[list(results.keys())[0]]['ds'].max(),
           alpha=0.03, color=C_GREEN)
ax.text(pd.Timestamp('2023-02-01'), feat['y'].max() * 0.97,
        '← History  |  Forecast →',
        color=TEXT_LO, fontsize=8, alpha=0.7)

# Plot each scenario
for name, cfg in SCENARIOS.items():
    df_sc = results[name]
    lw    = 1.5 if 'Baseline' in name else 2.5
    ms    = 0   if 'Baseline' in name else 5
    ax.plot(df_sc['ds'], df_sc['y_pred'],
            color=cfg['color'], lw=lw, linestyle=cfg['linestyle'],
            marker='o' if ms else '', markersize=ms,
            label=cfg['label'], alpha=0.9 if 'Baseline' not in name else 0.6)

# 2016 peak annotation
peak_idx  = feat['y'].idxmax()
peak_date = feat.loc[peak_idx, 'ds']
peak_val  = feat.loc[peak_idx, 'y']
ax.annotate('2016 Peak', xy=(peak_date, peak_val),
            xytext=(peak_date - pd.DateOffset(months=24), peak_val * 0.85),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
            color=C_RED, fontsize=8)

# Summary comparison box
base_total = results[list(SCENARIOS.keys())[0]]['y_pred'].sum()
lines = ["Scenario Comparison (18M Total Revenue)", "─"*38]
for name, cfg in SCENARIOS.items():
    total   = results[name]['y_pred'].sum()
    uplift  = (total - base_total) / base_total * 100
    tag     = f'{uplift:+.1f}%' if 'Baseline' not in name else 'baseline'
    sc_name = name.replace('\n', ' ')
    lines.append(f"{sc_name[:28]:28s}  {fmt_b(total):>8s}  {tag:>7s}")

# Prescriptive actions box
lines += ["", "─"*38, "Actions simulated:"]
for name, cfg in SCENARIOS.items():
    if 'action' in cfg:
        sc_short = name.split('\n')[0]
        lines.append(f"  {sc_short}: {cfg['action']}")

box_text = '\n'.join(lines)
ax.text(0.995, 0.97, box_text,
        transform=ax.transAxes, fontsize=7.5,
        va='top', ha='right', family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor=BG_MID,
                  edgecolor='#252A45', alpha=0.95),
        color=TEXT_HI)

ax.set_title(
    'PRESCRIPTIVE · TRAJECTORY 1  —  REVENUE FORECAST UNDER INTERVENTION SCENARIOS\n'
    'Root cause: Acquisition engine broken since 2016  |  '
    'Lever: Sessions (traffic) + Unique Visitors + Active Promos',
    fontsize=13, fontweight='bold', color=TEXT_HI, pad=14)
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Revenue (VND)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_b))
ax.legend(fontsize=8.5, framealpha=0.15, loc='upper left',
          ncol=1, handlelength=2)

plt.tight_layout()
plt.savefig('prescriptive_trajectory1.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("✅ prescriptive_trajectory1.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# 5. DETAILED SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  PRESCRIPTIVE SUMMARY — TRAJECTORY 1 INTERVENTION SCENARIOS")
print("="*70)

base_df    = results[list(SCENARIOS.keys())[0]]
base_total = base_df['y_pred'].sum()
base_avg   = base_df['y_pred'].mean()
base_peak  = base_df['y_pred'].max()

for name, cfg in SCENARIOS.items():
    df_sc   = results[name]
    total   = df_sc['y_pred'].sum()
    avg     = df_sc['y_pred'].mean()
    peak    = df_sc['y_pred'].max()
    uplift  = (total - base_total) / base_total * 100
    sc_name = name.replace('\n', ' ')

    print(f"\n{'─'*60}")
    print(f"  {sc_name}")
    if 'action' in cfg:
        print(f"  Action : {cfg['action']}")
    print(f"  18M Total Revenue : {fmt_b(total)}"
          + (f"  ({uplift:+.1f}% vs baseline)" if 'Baseline' not in name else ""))
    print(f"  Avg / Month       : {fmt_b(avg)}")
    print(f"  Peak Month        : {fmt_b(peak)}")

print(f"\n{'─'*60}")
print(f"  Additional revenue vs baseline:")
for name, cfg in SCENARIOS.items():
    if 'Baseline' not in name:
        df_sc  = results[name]
        uplift = df_sc['y_pred'].sum() - base_total
        sc_name = name.replace('\n', ' ')
        print(f"  {sc_name[:35]:35s}  +{fmt_b(uplift)}")







