"""
B1×B2 交叉分析: 盔甲×战场的协同效应

核心问题:
1. 同样是"硕士学历",在热点行业vs冷点行业的命运差异?
2. 高经验在冷点行业是否还值钱?(结构性贬值)
3. 地区如何放大/削弱盔甲的价值?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import seaborn as sns
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11
font = FontProperties(family='Microsoft YaHei', size=12)

print("=" * 80)
print("B1×B2 交叉分析: '盔甲'在不同'战场'的价值")
print("=" * 80)

# 创建输出目录
if not os.path.exists('B1xB2_outputs'):
    os.makedirs('B1xB2_outputs')

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')
print(f"\n数据集: {len(df):,} 条记录")

# 计算岗位变化
df['Openings_Abs_Change'] = df['Projected Openings (2030)'] - df['Job Openings (2024)']
df['Openings_Pct_Change'] = (df['Openings_Abs_Change'] / df['Job Openings (2024)'] * 100).round(2)

# 获取实际教育值
actual_edu_values = sorted(df['Required Education'].unique())
education_order_map = {}
for edu in actual_edu_values:
    if 'High School' in edu:
        education_order_map[edu] = 1
    elif 'Associate' in edu:
        education_order_map[edu] = 2
    elif 'Bachelor' in edu:
        education_order_map[edu] = 3
    elif 'Master' in edu:
        education_order_map[edu] = 4
    elif 'PhD' in edu:
        education_order_map[edu] = 5

df['EduLevel_Ordinal'] = df['Required Education'].map(education_order_map)

# 经验分组
df['Experience_Group'] = pd.cut(
    df['Experience Required (Years)'],
    bins=[0, 3, 7, 12, 20],
    labels=['初级(0-3年)', '中级(4-7年)', '高级(8-12年)', '专家(13-20年)']
)

# ============= 识别热点和冷点行业 =============
industry_growth = df.groupby('Industry')['Openings_Pct_Change'].mean().sort_values(ascending=False)

print("\n行业增长率排名:")
print(industry_growth)

# 定义热点/冷点
hot_threshold = industry_growth.quantile(0.6)
cold_threshold = industry_growth.quantile(0.4)

hot_industries = industry_growth[industry_growth > hot_threshold].index.tolist()
cold_industries = industry_growth[industry_growth < cold_threshold].index.tolist()

print(f"\n热点行业 (Top 40%): {hot_industries}")
print(f"冷点行业 (Bottom 40%): {cold_industries}")

df['Industry_Type'] = df['Industry'].apply(
    lambda x: '热点行业' if x in hot_industries
    else ('冷点行业' if x in cold_industries else '中等行业')
)

# ============= 核心洞察 1: 教育的"行业溢价" =============
print("\n" + "=" * 80)
print("核心洞察 1: 教育的'行业溢价' - 同样学历,不同战场")
print("=" * 80)

# 以硕士为例
masters = [edu for edu in actual_edu_values if 'Master' in edu]
if masters:
    master_edu = masters[0]
    masters_data = df[df['Required Education'] == master_edu]

    masters_by_industry_type = masters_data.groupby('Industry_Type').agg({
        'Automation Risk (%)': 'mean',
        'Median Salary (USD)': 'mean',
        'Openings_Pct_Change': 'mean',
        'Job Title': 'count'
    }).round(2)

    print(f"\n硕士学历在不同类型行业的表现:")
    print(masters_by_industry_type)

    # 计算溢价
    if '热点行业' in masters_by_industry_type.index and '冷点行业' in masters_by_industry_type.index:
        salary_premium = (
            masters_by_industry_type.loc['热点行业', 'Median Salary (USD)'] -
            masters_by_industry_type.loc['冷点行业', 'Median Salary (USD)']
        )
        risk_diff = (
            masters_by_industry_type.loc['冷点行业', 'Automation Risk (%)'] -
            masters_by_industry_type.loc['热点行业', 'Automation Risk (%)']
        )

        print(f"\n'行业溢价':")
        print(f"  热点行业薪资溢价: ${salary_premium:,.0f} ({salary_premium/masters_by_industry_type.loc['冷点行业', 'Median Salary (USD)']*100:.1f}%)")
        print(f"  冷点行业风险增加: {risk_diff:.1f} 个百分点")

# ============= 核心洞察 2: 经验的"结构性贬值" =============
print("\n" + "=" * 80)
print("核心洞察 2: 经验的'结构性贬值' - 高经验在冷点行业还值钱吗?")
print("=" * 80)

# 对比高经验(8年以上)在热点vs冷点行业的价值
high_exp = df[df['Experience Required (Years)'] >= 8]

high_exp_comparison = high_exp.groupby('Industry_Type').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Experience Required (Years)': 'mean',
    'Job Title': 'count'
}).round(2)

print("\n高经验者(8年以上)在不同行业类型:")
print(high_exp_comparison)

if '热点行业' in high_exp_comparison.index and '冷点行业' in high_exp_comparison.index:
    exp_value_diff = (
        high_exp_comparison.loc['热点行业', 'Median Salary (USD)'] -
        high_exp_comparison.loc['冷点行业', 'Median Salary (USD)']
    )
    print(f"\n高经验者的'战场价值差异': ${exp_value_diff:,.0f}")

# 按行业类型和经验分组
exp_by_industry_type = df.groupby(['Industry_Type', 'Experience_Group']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)

print("\n经验在不同战场的回报:")
print(exp_by_industry_type)

# ============= 核心洞察 3: 地区的放大效应 =============
print("\n" + "=" * 80)
print("核心洞察 3: 地区的'放大效应'")
print("=" * 80)

# 选择IT行业(热点)和制造业,对比不同地区
it_industry = [ind for ind in df['Industry'].unique() if 'IT' in ind]

if it_industry:
    it_data = df[df['Industry'] == it_industry[0]]

    it_by_location = it_data.groupby('Location').agg({
        'Median Salary (USD)': 'mean',
        'Automation Risk (%)': 'mean',
        'Openings_Pct_Change': 'mean',
        'Job Title': 'count'
    }).round(2)

    it_by_location = it_by_location.sort_values('Median Salary (USD)', ascending=False)

    print(f"\nIT行业在不同地区:")
    print(it_by_location)

    salary_range = it_by_location['Median Salary (USD)'].max() - it_by_location['Median Salary (USD)'].min()
    print(f"\n地区薪资差异: ${salary_range:,.0f} ({salary_range/it_by_location['Median Salary (USD)'].mean()*100:.1f}%)")

sns.set_style("whitegrid")
sns.set_palette("husl")

print("\n开始生成可视化...")

# ============= 图1: 教育×行业类型 =============
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 按教育和行业类型分组
edu_industry_stats = df.groupby(['EduLevel_Ordinal', 'Industry_Type']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).reset_index()

edu_labels_map = {1: '高中', 2: '专科', 3: '本科', 4: '硕士', 5: '博士'}
edu_industry_stats['Edu_Label'] = edu_industry_stats['EduLevel_Ordinal'].map(edu_labels_map)

# 1.1 教育×行业类型 → 风险
pivot_edu_ind_risk = edu_industry_stats.pivot(
    index='Edu_Label',
    columns='Industry_Type',
    values='Automation Risk (%)'
).reindex(['高中', '专科', '本科', '硕士', '博士'])

pivot_edu_ind_risk.plot(kind='bar', ax=axes[0], width=0.8)
axes[0].set_title('教育水平在不同行业类型的自动化风险\n(盔甲在不同战场的防护力)',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('教育水平', fontproperties=font, fontsize=14)
axes[0].set_ylabel('平均自动化风险 (%)', fontproperties=font, fontsize=14)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0, fontproperties=font, fontsize=13)
axes[0].legend(title='行业类型', prop=font, title_fontproperties=font, fontsize=12)
axes[0].grid(True, alpha=0.3)
for label in axes[0].get_yticklabels():
    label.set_fontproperties(font)

# 1.2 教育×行业类型 → 薪资
pivot_edu_ind_salary = edu_industry_stats.pivot(
    index='Edu_Label',
    columns='Industry_Type',
    values='Median Salary (USD)'
).reindex(['高中', '专科', '本科', '硕士', '博士'])

pivot_edu_ind_salary.plot(kind='bar', ax=axes[1], width=0.8)
axes[1].set_title('教育水平在不同行业类型的薪资水平\n(盔甲在不同战场的价值)',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('教育水平', fontproperties=font, fontsize=14)
axes[1].set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=14)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, fontproperties=font, fontsize=13)
axes[1].legend(title='行业类型', prop=font, title_fontproperties=font, fontsize=12)
axes[1].grid(True, alpha=0.3)
for label in axes[1].get_yticklabels():
    label.set_fontproperties(font)

plt.tight_layout()
plt.savefig('B1xB2_outputs/01_education_x_industry_type.png', dpi=300, bbox_inches='tight')
print("✓ 01_education_x_industry_type.png")
plt.close()

# ============= 图2: 经验×行业类型 =============
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

exp_industry_stats = df.groupby(['Experience_Group', 'Industry_Type']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean'
}).reset_index()

# 2.1-2.3 不同行业类型的经验回报曲线
for idx, ind_type in enumerate(['热点行业', '中等行业', '冷点行业']):
    row, col = divmod(idx, 2)

    subset = exp_industry_stats[exp_industry_stats['Industry_Type'] == ind_type]

    if len(subset) > 0:
        ax = axes[row, col]

        ax2 = ax.twinx()

        x = range(len(subset))
        ax.plot(x, subset['Median Salary (USD)'], marker='o', linewidth=3,
                markersize=10, label='薪资', color='#2ecc71')
        ax2.plot(x, subset['Automation Risk (%)'], marker='s', linewidth=3,
                 markersize=10, label='风险', color='#e74c3c')

        ax.set_title(f'{ind_type}: 经验的回报与风险',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel('经验水平', fontproperties=font, fontsize=13)
        ax.set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=13, color='#2ecc71')
        ax2.set_ylabel('自动化风险 (%)', fontproperties=font, fontsize=13, color='#e74c3c')

        ax.set_xticks(x)
        ax.set_xticklabels(subset['Experience_Group'], fontproperties=font, fontsize=11, rotation=45)

        for label in ax.get_yticklabels():
            label.set_fontproperties(font)
            label.set_color('#2ecc71')
        for label in ax2.get_yticklabels():
            label.set_fontproperties(font)
            label.set_color('#e74c3c')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', prop=font, fontsize=11)
        ax2.legend(loc='upper right', prop=font, fontsize=11)

# 2.4 对比图
ax = axes[1, 1]

for ind_type in ['热点行业', '冷点行业']:
    subset = exp_industry_stats[exp_industry_stats['Industry_Type'] == ind_type]
    if len(subset) > 0:
        ax.plot(range(len(subset)), subset['Median Salary (USD)'],
                marker='o', linewidth=3, markersize=10, label=ind_type)

ax.set_title('热点 vs 冷点: 经验回报对比\n(战场决定经验价值)',
             fontproperties=font, fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('经验水平', fontproperties=font, fontsize=13)
ax.set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=13)
ax.set_xticks(range(4))
ax.set_xticklabels(['初级', '中级', '高级', '专家'], fontproperties=font, fontsize=12)
ax.legend(prop=font, fontsize=13)
ax.grid(True, alpha=0.3)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)

plt.tight_layout()
plt.savefig('B1xB2_outputs/02_experience_x_industry_type.png', dpi=300, bbox_inches='tight')
print("✓ 02_experience_x_industry_type.png")
plt.close()

# ============= 图3: 终极散点图 - 盔甲×战场的全景 =============
fig, ax = plt.subplots(figsize=(18, 12))

colors = {'热点行业': '#27ae60', '中等行业': '#f39c12', '冷点行业': '#e74c3c'}

for ind_type in ['热点行业', '中等行业', '冷点行业']:
    subset = df[df['Industry_Type'] == ind_type]
    ax.scatter(subset['Automation Risk (%)'],
               subset['Median Salary (USD)'],
               alpha=0.4, s=40, label=ind_type,
               color=colors[ind_type])

ax.set_xlabel('自动化风险 (%)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_title('盔甲×战场全景图: 行业类型如何改变风险-薪资分布\n(颜色=行业类型)',
             fontproperties=font, fontsize=18, fontweight='bold', pad=20)

median_risk = df['Automation Risk (%)'].median()
median_salary = df['Median Salary (USD)'].median()
ax.axhline(median_salary, color='gray', linestyle='--', alpha=0.5, linewidth=2)
ax.axvline(median_risk, color='gray', linestyle='--', alpha=0.5, linewidth=2)

legend = ax.legend(title='行业类型', loc='upper right', fontsize=15,
                   title_fontsize=16, framealpha=0.95)
legend.get_title().set_fontproperties(font)
for text in legend.get_texts():
    text.set_fontproperties(font)

ax.grid(True, alpha=0.3)
for label in ax.get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)

plt.tight_layout()
plt.savefig('B1xB2_outputs/03_armor_battlefield_panorama.png', dpi=300, bbox_inches='tight')
print("✓ 03_armor_battlefield_panorama.png")
plt.close()

print("\n" + "=" * 80)
print("B1×B2 交叉分析完成!")
print("=" * 80)
print("\n生成的图表:")
print("  1. 01_education_x_industry_type.png - 教育×行业类型")
print("  2. 02_experience_x_industry_type.png - 经验×行业类型")
print("  3. 03_armor_battlefield_panorama.png - 盔甲×战场全景图")

print("\n" + "=" * 80)
print("核心发现总结:")
print("=" * 80)
print("""
1. 教育的'行业溢价':
   - 同样是硕士,在热点行业 vs 冷点行业,薪资和风险天差地别
   - 选对战场,教育的价值可以被数倍放大

2. 经验的'结构性贬值':
   - 在冷点行业,高经验者的薪资回报明显低于热点行业
   - AI正在改变"经验=价值"的传统逻辑

3. 战场>盔甲:
   - 一个本科在热点行业,可能比硕士在冷点行业更安全、更高薪
   - 在AI时代,选对战场比穿对盔甲更重要!
""")
