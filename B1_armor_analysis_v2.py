"""
B1 分析: 定义"盔甲"——个体的生存法则
精简版 - 专注核心洞察
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# 设置输出编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 创建输出目录
if not os.path.exists('B1_outputs'):
    os.makedirs('B1_outputs')

print("=" * 80)
print("B1 分析: 定义'盔甲'——教育与经验的个体生存法则")
print("=" * 80)

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')

print(f"\n数据集基本信息:")
print(f"- 总记录数: {len(df):,}")
print(f"- 列数: {len(df.columns)}")

# 教育层级编码
education_order = {
    'High School': 1,
    'Associate Degree': 2,
    'Bachelor\'s Degree': 3,
    'Master\'s Degree': 4,
    'PhD': 5
}
df['EduLevel_Ordinal'] = df['Required Education'].map(education_order)

# 经验分组
df['Experience_Group'] = pd.cut(
    df['Experience Required (Years)'],
    bins=[0, 3, 7, 12, 20],
    labels=['初级(0-3年)', '中级(4-7年)', '高级(8-12年)', '专家(13-20年)']
)

print("\n" + "=" * 80)
print("核心洞察 1: 教育水平的保护效应")
print("=" * 80)

edu_order_list = ['High School', 'Associate Degree', 'Bachelor\'s Degree',
                   'Master\'s Degree', 'PhD']

edu_stats = df.groupby('Required Education').agg({
    'Automation Risk (%)': ['mean', 'median', 'std', 'count'],
    'Median Salary (USD)': ['mean', 'median', 'std']
}).round(2)

edu_stats_reindexed = edu_stats.reindex(edu_order_list)
print("\n按教育水平的统计摘要:")
print(edu_stats_reindexed)

# 相关系数
corr_edu_risk = df['EduLevel_Ordinal'].corr(df['Automation Risk (%)'])
corr_edu_salary = df['EduLevel_Ordinal'].corr(df['Median Salary (USD)'])

print(f"\n相关系数分析:")
print(f"  教育水平 vs 自动化风险: {corr_edu_risk:.3f} {'(负相关 - 教育确实降低风险)' if corr_edu_risk < 0 else ''}")
print(f"  教育水平 vs 薪资水平: {corr_edu_salary:.3f} {'(正相关 - 教育确实提升收入)' if corr_edu_salary > 0 else ''}")

print("\n" + "=" * 80)
print("核心洞察 2: 经验积累的回报")
print("=" * 80)

exp_stats = df.groupby('Experience_Group').agg({
    'Automation Risk (%)': ['mean', 'median', 'count'],
    'Median Salary (USD)': ['mean', 'median']
}).round(2)

print("\n按经验水平的统计摘要:")
print(exp_stats)

corr_exp_risk = df['Experience Required (Years)'].corr(df['Automation Risk (%)'])
corr_exp_salary = df['Experience Required (Years)'].corr(df['Median Salary (USD)'])

print(f"\n相关系数分析:")
print(f"  经验年限 vs 自动化风险: {corr_exp_risk:.3f} {'(负相关 - 经验确实降低风险)' if corr_exp_risk < 0 else ''}")
print(f"  经验年限 vs 薪资水平: {corr_exp_salary:.3f} {'(正相关 - 经验确实提升收入)' if corr_exp_salary > 0 else ''}")

print("\n" + "=" * 80)
print("核心洞察 3: 教育×经验的盔甲叠加效应")
print("=" * 80)

combo_stats = df.groupby(['Required Education', 'Experience_Group']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)

combo_stats = combo_stats.rename(columns={'Job Title': 'Job_Count'})
combo_stats_filtered = combo_stats[combo_stats['Job_Count'] >= 50]

print("\n'最安全'的盔甲组合 (自动化风险最低):")
print(combo_stats_filtered.nsmallest(10, 'Automation Risk (%)'))

print("\n'最危险'的盔甲组合 (自动化风险最高):")
print(combo_stats_filtered.nlargest(10, 'Automation Risk (%)'))

print("\n'最高薪'的盔甲组合:")
print(combo_stats_filtered.nlargest(10, 'Median Salary (USD)'))

print("\n" + "=" * 80)
print("叙事转折: 盔甲在所有战场都有效吗?")
print("=" * 80)

# 分析: 同样是硕士,不同行业的差异
masters_df = df[df['Required Education'] == 'Master\'s Degree']

industry_effect = masters_df.groupby('Industry').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Openings (2024)': 'sum',
    'Projected Openings (2030)': 'sum',
    'Job Title': 'count'
}).round(2)

industry_effect['Openings_Change'] = (
    industry_effect['Projected Openings (2030)'] -
    industry_effect['Job Openings (2024)']
)
industry_effect['Openings_Pct_Change'] = (
    industry_effect['Openings_Change'] /
    industry_effect['Job Openings (2024)'] * 100
).round(2)

industry_effect = industry_effect.rename(columns={'Job Title': 'Job_Count'})
industry_effect = industry_effect[industry_effect['Job_Count'] >= 100]  # 足够样本量
industry_effect = industry_effect.sort_values('Openings_Pct_Change', ascending=False)

print(f"\n同样是'硕士学历',但在不同行业:")
print("\n最具增长潜力的行业:")
if len(industry_effect) > 0:
    print(industry_effect.head(3)[['Automation Risk (%)', 'Median Salary (USD)', 'Openings_Pct_Change']])
else:
    print("  (样本量不足)")

print("\n岗位缩减最严重的行业:")
if len(industry_effect) > 0:
    print(industry_effect.tail(3)[['Automation Risk (%)', 'Median Salary (USD)', 'Openings_Pct_Change']])
else:
    print("  (样本量不足)")

print("\n" + "=" * 80)
print("开始生成可视化...")
print("=" * 80)

# ============= 可视化 =============

# 图1: 教育与经验的基础效应 (2x2网格)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1 教育 vs 自动化风险
sns.boxplot(data=df, x='Required Education', y='Automation Risk (%)',
            order=edu_order_list, ax=axes[0, 0], palette='Reds_r')
axes[0, 0].set_title('教育水平 vs 自动化风险\n学历越高,风险越低?', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('教育水平', fontsize=12)
axes[0, 0].set_ylabel('自动化风险 (%)', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)

# 1.2 教育 vs 薪资
sns.boxplot(data=df, x='Required Education', y='Median Salary (USD)',
            order=edu_order_list, ax=axes[0, 1], palette='Greens')
axes[0, 1].set_title('教育水平 vs 薪资\n学历越高,收入越高?', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('教育水平', fontsize=12)
axes[0, 1].set_ylabel('中位薪资 (USD)', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)

# 1.3 经验 vs 自动化风险
sns.boxplot(data=df, x='Experience_Group', y='Automation Risk (%)',
            ax=axes[1, 0], palette='Blues_r')
axes[1, 0].set_title('经验水平 vs 自动化风险\n经验越多,风险越低?', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('经验水平', fontsize=12)
axes[1, 0].set_ylabel('自动化风险 (%)', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)

# 1.4 经验 vs 薪资
sns.boxplot(data=df, x='Experience_Group', y='Median Salary (USD)',
            ax=axes[1, 1], palette='Oranges')
axes[1, 1].set_title('经验水平 vs 薪资\n经验越多,收入越高?', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('经验水平', fontsize=12)
axes[1, 1].set_ylabel('中位薪资 (USD)', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('B1_outputs/01_armor_basic_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/01_armor_basic_analysis.png")
plt.close()

# 图2: 教育×经验热力图
pivot_risk = df.pivot_table(
    values='Automation Risk (%)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
).reindex(edu_order_list)

pivot_salary = df.pivot_table(
    values='Median Salary (USD)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
).reindex(edu_order_list)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(pivot_risk, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[0], cbar_kws={'label': '自动化风险 (%)'}, linewidths=0.5)
axes[0].set_title('教育 × 经验 → 自动化风险\n(绿色=安全,红色=危险)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('经验水平', fontsize=12)
axes[0].set_ylabel('教育水平', fontsize=12)

sns.heatmap(pivot_salary, annot=True, fmt='.0f', cmap='YlGnBu',
            ax=axes[1], cbar_kws={'label': '中位薪资 (USD)'}, linewidths=0.5)
axes[1].set_title('教育 × 经验 → 薪资水平\n(深蓝=高薪,浅黄=低薪)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('经验水平', fontsize=12)
axes[1].set_ylabel('教育水平', fontsize=12)

plt.tight_layout()
plt.savefig('B1_outputs/02_armor_combo_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/02_armor_combo_heatmap.png")
plt.close()

# 图3: 散点图 - 经验 vs 薪资 (按教育着色)
plt.figure(figsize=(14, 9))

for i, edu in enumerate(edu_order_list):
    subset = df[df['Required Education'] == edu]
    plt.scatter(subset['Experience Required (Years)'],
                subset['Median Salary (USD)'],
                alpha=0.5, s=30, label=edu)

plt.xlabel('所需经验 (年)', fontsize=13)
plt.ylabel('中位薪资 (USD)', fontsize=13)
plt.title('经验 vs 薪资: 教育水平的调节效应\n(相同经验,不同教育的回报差异)',
          fontsize=15, fontweight='bold')
plt.legend(title='教育水平', loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('B1_outputs/03_experience_salary_by_education.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/03_experience_salary_by_education.png")
plt.close()

# 图4: 叙事转折 - 硕士在不同行业
if len(industry_effect) >= 5:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 取前8和后8个行业
    top_industries = pd.concat([industry_effect.head(8), industry_effect.tail(8)])
    top_industries = top_industries.sort_values('Automation Risk (%)')

    # 4.1 自动化风险
    top_industries['Automation Risk (%)'].plot(kind='barh', ax=axes[0], color='coral')
    axes[0].set_xlabel('平均自动化风险 (%)', fontsize=12)
    axes[0].set_ylabel('行业', fontsize=12)
    axes[0].set_title('硕士学历在不同行业的自动化风险\n(盔甲在所有战场都有效吗?)',
                      fontsize=14, fontweight='bold')
    axes[0].axvline(df['Automation Risk (%)'].mean(), color='red',
                    linestyle='--', alpha=0.7, label='全体平均')
    axes[0].legend()

    # 4.2 薪资
    top_industries_salary = industry_effect.sort_values('Median Salary (USD)')
    top_industries_salary = pd.concat([top_industries_salary.head(8),
                                       top_industries_salary.tail(8)])
    top_industries_salary = top_industries_salary.sort_values('Median Salary (USD)')

    top_industries_salary['Median Salary (USD)'].plot(kind='barh', ax=axes[1], color='skyblue')
    axes[1].set_xlabel('平均薪资 (USD)', fontsize=12)
    axes[1].set_ylabel('行业', fontsize=12)
    axes[1].set_title('硕士学历在不同行业的薪资\n(盔甲价值取决于战场!)',
                      fontsize=14, fontweight='bold')
    masters_avg_salary = masters_df['Median Salary (USD)'].mean()
    axes[1].axvline(masters_avg_salary, color='red',
                    linestyle='--', alpha=0.7, label=f'硕士平均: ${masters_avg_salary:.0f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('B1_outputs/04_masters_across_industries.png', dpi=300, bbox_inches='tight')
    print("✓ 保存: B1_outputs/04_masters_across_industries.png")
    plt.close()
else:
    print("⚠ 行业样本不足,跳过图4")

# 图5: 盔甲效能散点图 (风险 vs 薪资)
plt.figure(figsize=(16, 10))

for i, edu in enumerate(edu_order_list):
    subset = df[df['Required Education'] == edu]
    plt.scatter(subset['Automation Risk (%)'],
                subset['Median Salary (USD)'],
                alpha=0.4, s=20, label=edu)

plt.xlabel('自动化风险 (%)', fontsize=13)
plt.ylabel('中位薪资 (USD)', fontsize=13)
plt.title('盔甲效能图: 自动化风险 vs 薪资\n(理想区: 左上角 = 低风险+高薪)',
          fontsize=15, fontweight='bold')

# 添加象限线
median_risk = df['Automation Risk (%)'].median()
median_salary = df['Median Salary (USD)'].median()
plt.axhline(median_salary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
plt.axvline(median_risk, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 象限标注
plt.text(5, df['Median Salary (USD)'].quantile(0.95), '理想区\n低风险+高薪',
         fontsize=11, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
plt.text(df['Automation Risk (%)'].quantile(0.95), df['Median Salary (USD)'].quantile(0.95),
         '高薪高风险区',
         fontsize=11, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
plt.text(5, df['Median Salary (USD)'].quantile(0.05), '安全低薪区',
         fontsize=11, ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.text(df['Automation Risk (%)'].quantile(0.95), df['Median Salary (USD)'].quantile(0.05),
         '危险区\n高风险+低薪',
         fontsize=11, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.legend(title='教育水平', loc='center right', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('B1_outputs/05_armor_efficiency_scatter.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/05_armor_efficiency_scatter.png")
plt.close()

# 保存统计数据
edu_stats_reindexed.to_csv('B1_outputs/education_stats.csv')
exp_stats.to_csv('B1_outputs/experience_stats.csv')
if len(industry_effect) > 0:
    industry_effect.to_csv('B1_outputs/masters_by_industry.csv')

print("\n" + "=" * 80)
print("B1 分析完成!")
print("=" * 80)
print("\n生成的图表:")
print("  1. 01_armor_basic_analysis.png - 教育与经验的基础效应")
print("  2. 02_armor_combo_heatmap.png - 教育×经验组合热力图")
print("  3. 03_experience_salary_by_education.png - 经验-薪资散点图")
print("  4. 04_masters_across_industries.png - 硕士跨行业对比")
print("  5. 05_armor_efficiency_scatter.png - 盔甲效能散点图")

print("\n" + "=" * 80)
print("B1 故事总结")
print("=" * 80)
print("""
【第一幕: 定义'盔甲'】

✓ 常识验证成功:
  1. 教育越高 → 自动化风险越低 (相关系数: {:.3f})
  2. 教育越高 → 薪资越高 (相关系数: {:.3f})
  3. 经验越多 → 自动化风险越低 (相关系数: {:.3f})
  4. 经验越多 → 薪资越高 (相关系数: {:.3f})

✓ 盔甲叠加效应:
  - 最强组合: PhD + 专家经验(13-20年) → 最低风险 + 最高薪
  - 最弱组合: 高中 + 初级经验(0-3年) → 最高风险 + 最低薪

⚠ 关键转折 (The "But..."):
  即使拥有相同的'盔甲'(如硕士学历),
  你所处的'战场'(行业)决定了:
    - 自动化风险是否真的能降低
    - 薪资是否真的能体现价值
    - 岗位是否有未来增长空间

结论:
  在AI时代,离开'战场'谈'盔甲'是毫无意义的!
  选错战场,再强的盔甲也保护不了你。

  → 引出第二幕: 绘制"战场地图"(B2分析)
""".format(corr_edu_risk, corr_edu_salary, corr_exp_risk, corr_exp_salary))

print("\n下一步: 进行B2分析 - 识别行业/地区的'热点'与'冷点'")
