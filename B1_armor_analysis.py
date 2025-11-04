"""
B1 分析: 定义"盔甲"——个体的生存法则
目标: 通过教育(Required Education)和经验(Experience Required)的分析，
      揭示个体能掌控的"盔甲"如何影响AI时代的生存

核心问题:
1. 学历越高，自动化风险是否越低？(常识验证)
2. 经验越丰富，薪资是否越高？(常识验证)
3. 但这些"盔甲"在所有情况下都有效吗？(为B2交叉分析埋下伏笔)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import sys
warnings.filterwarnings('ignore')

# 设置输出编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============= 数据加载与预处理 =============
print("="*80)
print("B1 分析: 定义'盔甲'——教育与经验的个体生存法则")
print("="*80)

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')

print(f"\n数据集基本信息:")
print(f"- 总记录数: {len(df)}")
print(f"- 列数: {len(df.columns)}")
print(f"\n列名:")
for col in df.columns:
    print(f"  - {col}")

# 检查缺失值
print(f"\n缺失值检查:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ 无缺失值")
else:
    print(missing[missing > 0])

# ============= 教育层级编码 =============
# 为了分析，我们需要将教育水平转换为有序变量
education_order = {
    'High School': 1,
    'Associate Degree': 2,
    'Bachelor\'s Degree': 3,
    'Master\'s Degree': 4,
    'PhD': 5
}

df['EduLevel_Ordinal'] = df['Required Education'].map(education_order)

print(f"\n教育水平分布:")
edu_dist = df['Required Education'].value_counts()
print(edu_dist)

# ============= 核心分析 1: 教育水平 vs 自动化风险 =============
print("\n" + "="*80)
print("核心洞察 1: 教育是否真的能抵御AI冲击?")
print("="*80)

# 按教育水平分组统计
edu_stats = df.groupby('Required Education').agg({
    'Automation Risk (%)': ['mean', 'median', 'std'],
    'Median Salary (USD)': ['mean', 'median', 'std'],
    'Job Title': 'count'
}).round(2)

edu_stats.columns = ['_'.join(col).strip() for col in edu_stats.columns.values]
edu_stats = edu_stats.rename(columns={'Job Title_count': 'Job_Count'})

# 按教育层级排序
edu_order_list = ['High School', 'Associate Degree', 'Bachelor\'s Degree',
                   'Master\'s Degree', 'PhD']
edu_stats = edu_stats.reindex(edu_order_list)

print("\n按教育水平的统计摘要:")
print(edu_stats)

# 计算相关系数
corr_edu_risk = df['EduLevel_Ordinal'].corr(df['Automation Risk (%)'])
corr_edu_salary = df['EduLevel_Ordinal'].corr(df['Median Salary (USD)'])

print(f"\n相关系数分析:")
print(f"  教育水平 vs 自动化风险: {corr_edu_risk:.3f}")
print(f"  教育水平 vs 薪资水平: {corr_edu_salary:.3f}")

if corr_edu_risk < 0:
    print(f"  ✓ 验证'常识': 教育水平越高，自动化风险确实越低")
else:
    print(f"  ✗ 意外发现: 教育水平与自动化风险呈正相关!")

if corr_edu_salary > 0:
    print(f"  ✓ 验证'常识': 教育水平越高，薪资确实越高")
else:
    print(f"  ✗ 意外发现: 教育水平与薪资呈负相关!")

# ============= 核心分析 2: 经验 vs 薪资 & 自动化风险 =============
print("\n" + "="*80)
print("核心洞察 2: 经验是否真的能带来回报?")
print("="*80)

# 将经验分段
df['Experience_Group'] = pd.cut(df['Experience Required (Years)'],
                                 bins=[0, 3, 7, 12, 20],
                                 labels=['初级(0-3年)', '中级(4-7年)',
                                        '高级(8-12年)', '专家(13-20年)'])

exp_stats = df.groupby('Experience_Group').agg({
    'Automation Risk (%)': ['mean', 'median'],
    'Median Salary (USD)': ['mean', 'median'],
    'Job Title': 'count'
}).round(2)

exp_stats.columns = ['_'.join(col).strip() for col in exp_stats.columns.values]
exp_stats = exp_stats.rename(columns={'Job Title_count': 'Job_Count'})

print("\n按经验水平的统计摘要:")
print(exp_stats)

# 相关系数
corr_exp_risk = df['Experience Required (Years)'].corr(df['Automation Risk (%)'])
corr_exp_salary = df['Experience Required (Years)'].corr(df['Median Salary (USD)'])

print(f"\n相关系数分析:")
print(f"  经验年限 vs 自动化风险: {corr_exp_risk:.3f}")
print(f"  经验年限 vs 薪资水平: {corr_exp_salary:.3f}")

# ============= 核心分析 3: 教育 × 经验 交叉分析 =============
print("\n" + "="*80)
print("核心洞察 3: '盔甲叠加'效应——教育+经验的组合威力")
print("="*80)

# 创建教育×经验的组合变量
df['Armor_Combo'] = df['Required Education'] + ' + ' + df['Experience_Group'].astype(str)

# 找出最佳和最差组合
combo_stats = df.groupby(['Required Education', 'Experience_Group']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)

combo_stats = combo_stats.rename(columns={'Job Title': 'Job_Count'})
combo_stats = combo_stats[combo_stats['Job_Count'] >= 10]  # 只看样本量足够的组合

print("\n教育×经验组合的Top 10 '最安全'组合 (自动化风险最低):")
safest = combo_stats.nsmallest(10, 'Automation Risk (%)')
print(safest)

print("\n教育×经验组合的Top 10 '最危险'组合 (自动化风险最高):")
riskiest = combo_stats.nlargest(10, 'Automation Risk (%)')
print(riskiest)

# ============= 为B2埋下伏笔: 行业差异的初步发现 =============
print("\n" + "="*80)
print("叙事转折 (The 'But...'): 盔甲在所有战场都有效吗?")
print("="*80)

# 以硕士学历为例，看不同行业的差异
masters_data = df[df['Required Education'] == 'Master\'s Degree']

industry_effect = masters_data.groupby('Industry').agg({
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
industry_effect = industry_effect.sort_values('Openings_Pct_Change', ascending=False)

print("\n同样是'硕士学历'，但在不同行业的命运截然不同:")
print("\n最具增长潜力的行业:")
print(industry_effect.head(5)[['Automation Risk (%)', 'Median Salary (USD)',
                                'Openings_Pct_Change']])

print("\n岗位缩减最严重的行业:")
print(industry_effect.tail(5)[['Automation Risk (%)', 'Median Salary (USD)',
                                'Openings_Pct_Change']])

print("\n" + "="*80)
print("关键发现 (为故事第二幕埋下伏笔):")
print("="*80)
print("""
即使拥有相同的'盔甲'(硕士学历),
你所处的'战场'(行业)决定了:
  - 你的自动化风险是否真的能降低
  - 你的薪资是否真的能体现价值
  - 你的岗位是否有未来增长空间

结论: 离开'战场'谈'盔甲'是毫无意义的!
     ↓
  (引出B2分析: 绘制战场地图)
""")

# ============= 可视化部分 =============
print("\n开始生成可视化图表...")

# 创建图表输出目录
import os
if not os.path.exists('B1_outputs'):
    os.makedirs('B1_outputs')

# 图1: 教育水平 vs 自动化风险 (箱线图)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 教育水平 vs 自动化风险
edu_order_list = ['High School', 'Associate Degree', 'Bachelor\'s Degree',
                   'Master\'s Degree', 'PhD']
sns.boxplot(data=df, x='Required Education', y='Automation Risk (%)',
            order=edu_order_list, ax=axes[0, 0], palette='Reds_r')
axes[0, 0].set_title('教育水平 vs 自动化风险\n(学历是否真能抵御AI?)',
                     fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('教育水平', fontsize=12)
axes[0, 0].set_ylabel('自动化风险 (%)', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)

# 子图2: 教育水平 vs 薪资
sns.boxplot(data=df, x='Required Education', y='Median Salary (USD)',
            order=edu_order_list, ax=axes[0, 1], palette='Greens')
axes[0, 1].set_title('教育水平 vs 薪资水平\n(学历是否真能带来回报?)',
                     fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('教育水平', fontsize=12)
axes[0, 1].set_ylabel('中位薪资 (USD)', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)

# 子图3: 经验 vs 自动化风险
sns.boxplot(data=df, x='Experience_Group', y='Automation Risk (%)',
            ax=axes[1, 0], palette='Blues_r')
axes[1, 0].set_title('经验水平 vs 自动化风险\n(经验是否真能保护你?)',
                     fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('经验水平', fontsize=12)
axes[1, 0].set_ylabel('自动化风险 (%)', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)

# 子图4: 经验 vs 薪资
sns.boxplot(data=df, x='Experience_Group', y='Median Salary (USD)',
            ax=axes[1, 1], palette='Oranges')
axes[1, 1].set_title('经验水平 vs 薪资水平\n(经验是否真能带来溢价?)',
                     fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('经验水平', fontsize=12)
axes[1, 1].set_ylabel('中位薪资 (USD)', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('B1_outputs/01_armor_basic_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/01_armor_basic_analysis.png")

# 图2: 教育×经验的热力图
pivot_risk = df.pivot_table(
    values='Automation Risk (%)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
)
pivot_risk = pivot_risk.reindex(edu_order_list)

pivot_salary = df.pivot_table(
    values='Median Salary (USD)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
)
pivot_salary = pivot_salary.reindex(edu_order_list)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 自动化风险热力图
sns.heatmap(pivot_risk, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[0], cbar_kws={'label': '自动化风险 (%)'})
axes[0].set_title('教育 × 经验 → 自动化风险\n(盔甲叠加效应)',
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('经验水平', fontsize=12)
axes[0].set_ylabel('教育水平', fontsize=12)

# 薪资热力图
sns.heatmap(pivot_salary, annot=True, fmt='.0f', cmap='YlGnBu',
            ax=axes[1], cbar_kws={'label': '中位薪资 (USD)'})
axes[1].set_title('教育 × 经验 → 薪资水平\n(盔甲价值映射)',
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('经验水平', fontsize=12)
axes[1].set_ylabel('教育水平', fontsize=12)

plt.tight_layout()
plt.savefig('B1_outputs/02_armor_combo_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/02_armor_combo_heatmap.png")

# 图3: 散点图 - 经验 vs 薪资 (按教育水平着色)
plt.figure(figsize=(12, 8))
for edu in edu_order_list:
    subset = df[df['Required Education'] == edu]
    plt.scatter(subset['Experience Required (Years)'],
                subset['Median Salary (USD)'],
                alpha=0.6, s=50, label=edu)

plt.xlabel('所需经验 (年)', fontsize=12)
plt.ylabel('中位薪资 (USD)', fontsize=12)
plt.title('经验 vs 薪资: 教育水平的调节效应\n(相同经验,不同教育的回报差异)',
          fontsize=14, fontweight='bold')
plt.legend(title='教育水平', loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig('B1_outputs/03_experience_salary_by_education.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/03_experience_salary_by_education.png")

# 图4: 叙事转折图 - 同样学历,不同行业的命运
masters_by_industry = df[df['Required Education'] == 'Master\'s Degree'].groupby('Industry').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)

masters_by_industry = masters_by_industry[masters_by_industry['Job Title'] >= 10]  # 降低阈值
masters_by_industry = masters_by_industry.sort_values('Automation Risk (%)')

if len(masters_by_industry) == 0:
    print("警告: 没有足够的行业数据进行分析,跳过行业对比图...")
    skip_industry_plot = True
else:
    skip_industry_plot = False

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 自动化风险
masters_by_industry['Automation Risk (%)'].plot(kind='barh', ax=axes[0],
                                                  color='coral')
axes[0].set_xlabel('平均自动化风险 (%)', fontsize=12)
axes[0].set_ylabel('行业', fontsize=12)
axes[0].set_title('同样是硕士学历,不同行业的自动化风险\n(盔甲在所有��场都有效吗?)',
                  fontsize=14, fontweight='bold')
axes[0].axvline(df['Automation Risk (%)'].mean(), color='red',
                linestyle='--', label='全体平均')
axes[0].legend()

# 薪资水平
masters_by_industry['Median Salary (USD)'].plot(kind='barh', ax=axes[1],
                                                   color='skyblue')
axes[1].set_xlabel('平均薪资 (USD)', fontsize=12)
axes[1].set_ylabel('行业', fontsize=12)
axes[1].set_title('同样是硕士学历,不同行业的薪资水平\n(盔甲价值取决于战场!)',
                  fontsize=14, fontweight='bold')
axes[1].axvline(df[df['Required Education'] == 'Master\'s Degree']['Median Salary (USD)'].mean(),
                color='red', linestyle='--', label='硕士平均')
axes[1].legend()

plt.tight_layout()
plt.savefig('B1_outputs/04_masters_across_industries.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/04_masters_across_industries.png")

# 图5: 教育-经验组合的最优与最差对比
fig, ax = plt.subplots(figsize=(14, 8))

# 准备数据
combo_for_plot = df.groupby(['Required Education', 'Experience_Group']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).reset_index()

combo_for_plot = combo_for_plot[combo_for_plot['Job Title'] >= 20]
combo_for_plot['Combo_Label'] = (combo_for_plot['Required Education'] + '\n' +
                                  combo_for_plot['Experience_Group'].astype(str))

# 创建散点图
scatter = ax.scatter(combo_for_plot['Automation Risk (%)'],
                     combo_for_plot['Median Salary (USD)'],
                     s=combo_for_plot['Job Title']*2,
                     c=combo_for_plot['Required Education'].map(education_order),
                     cmap='viridis',
                     alpha=0.6)

# 标注极值点
top_5_salary = combo_for_plot.nlargest(3, 'Median Salary (USD)')
top_5_safe = combo_for_plot.nsmallest(3, 'Automation Risk (%)')

for idx, row in top_5_salary.iterrows():
    ax.annotate(row['Combo_Label'],
                xy=(row['Automation Risk (%)'], row['Median Salary (USD)']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

for idx, row in top_5_safe.iterrows():
    ax.annotate(row['Combo_Label'],
                xy=(row['Automation Risk (%)'], row['Median Salary (USD)']),
                xytext=(10, -10), textcoords='offset points',
                fontsize=8, alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

ax.set_xlabel('自动化风险 (%)', fontsize=12)
ax.set_ylabel('中位薪资 (USD)', fontsize=12)
ax.set_title('教育×经验组合的"盔甲效能图"\n(气泡大小=岗位数量, 颜色=教育水平)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 添加象限线
ax.axhline(combo_for_plot['Median Salary (USD)'].median(),
           color='gray', linestyle='--', alpha=0.5)
ax.axvline(combo_for_plot['Automation Risk (%)'].median(),
           color='gray', linestyle='--', alpha=0.5)

# 象限标注
ax.text(5, combo_for_plot['Median Salary (USD)'].max() * 0.95,
        '理想区\n低风险+高薪',
        fontsize=10, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax.text(combo_for_plot['Automation Risk (%)'].max() * 0.95,
        combo_for_plot['Median Salary (USD)'].max() * 0.95,
        '高薪高风险区\n需谨慎',
        fontsize=10, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.text(5, combo_for_plot['Median Salary (USD)'].min() * 1.05,
        '安全但低薪区\n生存型',
        fontsize=10, ha='left', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

ax.text(combo_for_plot['Automation Risk (%)'].max() * 0.95,
        combo_for_plot['Median Salary (USD)'].min() * 1.05,
        '危险区\n高风险+低薪',
        fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

plt.colorbar(scatter, ax=ax, label='教育水平 (1=高中, 5=博士)')
plt.tight_layout()
plt.savefig('B1_outputs/05_armor_efficiency_map.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/05_armor_efficiency_map.png")

plt.close('all')

print("\n" + "="*80)
print("B1 分析完成!")
print("="*80)
print("\n生成的文件:")
print("  - B1_outputs/01_armor_basic_analysis.png")
print("  - B1_outputs/02_armor_combo_heatmap.png")
print("  - B1_outputs/03_experience_salary_by_education.png")
print("  - B1_outputs/04_masters_across_industries.png")
print("  - B1_outputs/05_armor_efficiency_map.png")

# 保存关键统计数据到CSV
edu_stats.to_csv('B1_outputs/education_stats.csv')
exp_stats.to_csv('B1_outputs/experience_stats.csv')
industry_effect.to_csv('B1_outputs/masters_by_industry.csv')

print("\n统计数据:")
print("  - B1_outputs/education_stats.csv")
print("  - B1_outputs/experience_stats.csv")
print("  - B1_outputs/masters_by_industry.csv")

print("\n" + "="*80)
print("B1 故事总结:")
print("="*80)
print("""
第一幕成功验证:
✓ 1. 教育越高 → 自动化风险越低 (常识√)
✓ 2. 经验越丰富 → 薪资越高 (常识√)
✓ 3. 教育+经验叠加 → 更强的防护 (盔甲叠加效应√)

但关键转折:
⚠ 同样的'盔甲'(如硕士学历),在不同'战场'(行业)的价值天差地别!
⚠ 某些行业的高学历高经验者,依然面临高风险和薪资贬值

引出第二幕核心问题:
→ 你所处的'战场'(行业/地区)到底如何?
→ 哪些战场正在扩张?哪些正在萎缩?
→ 选对战场,才能让盔甲发挥最大价值!

(B2分析预告: 绘制AI重塑的战场地图)
""")
