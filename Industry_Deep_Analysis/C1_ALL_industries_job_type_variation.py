"""
C1补充分析: 所有行业的职位类型风险变异分析
========================================

核心问题:
1. IT的小变异(1.59%)是个例还是普遍现象?
2. 每个行业内部,不同职位类型的风险差异有多大?
3. 是"一荣俱荣,一损俱损"还是"内部分化"?
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
print("C1补充分析: 所有行业职位类型风险变异分析")
print("=" * 80)

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')

# 职位类型分类函数 (与C1一致)
def classify_job_type(title):
    title_lower = str(title).lower()

    if any(word in title_lower for word in ['manager', 'director', 'executive', 'head', 'chief', 'president']):
        return 'Management'
    elif any(word in title_lower for word in ['engineer', 'developer', 'programmer', 'architect', 'scientist']):
        return 'Engineering'
    elif any(word in title_lower for word in ['administrator', 'coordinator', 'assistant', 'clerk', 'secretary']):
        return 'Administrative'
    elif any(word in title_lower for word in ['doctor', 'physician', 'nurse', 'therapist', 'medical', 'health']):
        return 'Medical_Professional'
    elif any(word in title_lower for word in ['teacher', 'professor', 'instructor', 'educator', 'trainer']):
        return 'Education'
    elif any(word in title_lower for word in ['designer', 'artist', 'writer', 'creative', 'photographer']):
        return 'Creative'
    elif any(word in title_lower for word in ['sales', 'marketing', 'business development']):
        return 'Sales_Marketing'
    elif any(word in title_lower for word in ['technician', 'operator', 'mechanic', 'driver', 'pilot']):
        return 'Technical_Operator'
    elif any(word in title_lower for word in ['analyst', 'researcher', 'consultant', 'advisor']):
        return 'Analysis'
    else:
        return 'Other'

df['Job_Type'] = df['Job Title'].apply(classify_job_type)

# ============= 分析1: 每个行业的职位类型风险分布 =============
print("\n" + "=" * 80)
print("分析1: 每个行业内部职位类型的风险变异")
print("=" * 80)

industries = df['Industry'].unique()
industry_variation_summary = []

for industry in sorted(industries):
    industry_data = df[df['Industry'] == industry]

    # 按职位类型统计 (样本量>30)
    job_type_stats = industry_data.groupby('Job_Type').agg({
        'Automation Risk (%)': ['mean', 'count'],
        'Median Salary (USD)': 'mean'
    }).round(2)

    job_type_stats.columns = ['Risk_Mean', 'Count', 'Salary_Mean']
    job_type_stats = job_type_stats[job_type_stats['Count'] >= 30]

    if len(job_type_stats) > 0:
        risk_min = job_type_stats['Risk_Mean'].min()
        risk_max = job_type_stats['Risk_Mean'].max()
        risk_range = risk_max - risk_min
        risk_std = job_type_stats['Risk_Mean'].std()
        risk_cv = risk_std / job_type_stats['Risk_Mean'].mean() if job_type_stats['Risk_Mean'].mean() > 0 else 0

        industry_variation_summary.append({
            'Industry': industry,
            'Risk_Min': risk_min,
            'Risk_Max': risk_max,
            'Risk_Range': risk_range,
            'Risk_Std': risk_std,
            'Risk_CV': risk_cv,
            'Job_Types_Count': len(job_type_stats),
            'Total_Jobs': len(industry_data)
        })

        print(f"\n【{industry}】")
        print(f"  职位类型数量: {len(job_type_stats)}")
        print(f"  风险范围: {risk_min:.2f}% - {risk_max:.2f}%")
        print(f"  风险极差: {risk_range:.2f}个百分点")
        print(f"  风险标准差: {risk_std:.2f}")
        print(f"  变异系数(CV): {risk_cv:.4f}")
        print(f"  详细数据:")
        for job_type, row in job_type_stats.sort_values('Risk_Mean').iterrows():
            print(f"    - {job_type:20s}: {row['Risk_Mean']:.2f}% (n={int(row['Count']):,}, ${row['Salary_Mean']:,.0f})")

# 创建汇总DataFrame
variation_df = pd.DataFrame(industry_variation_summary)
variation_df = variation_df.sort_values('Risk_Range', ascending=False)

print("\n" + "=" * 80)
print("汇总: 各行业内部风险变异排名")
print("=" * 80)
print(variation_df.to_string(index=False))

variation_df.to_csv('Industry_Deep_Analysis/C1_outputs/industry_internal_variation_summary.csv', index=False)

# ============= 分析2: 判断是"一荣俱荣"还是"分化" =============
print("\n" + "=" * 80)
print("分析2: 判定 - 一荣俱荣 vs 内部分化")
print("=" * 80)

# 定义阈值 (基于数据特征)
LOW_VARIATION_THRESHOLD = 3.0  # 极差<3%为"低变异"
MODERATE_VARIATION_THRESHOLD = 6.0  # 极差3-6%为"中等变异"

for _, row in variation_df.iterrows():
    industry = row['Industry']
    risk_range = row['Risk_Range']

    if risk_range < LOW_VARIATION_THRESHOLD:
        pattern = "✅ 一荣俱荣,一损俱损 (内部高度统一)"
        color = "绿色"
    elif risk_range < MODERATE_VARIATION_THRESHOLD:
        pattern = "⚠️ 轻度分化 (有差异但不极端)"
        color = "黄色"
    else:
        pattern = "❌ 显著分化 (内部差异巨大)"
        color = "红色"

    print(f"{industry:15s} | 极差={risk_range:5.2f}% | {pattern}")

# ============= 可视化部分 =============
print("\n开始生成可视化...")

# 图1: 所有行业的风险变异对比 (2×2)
fig, axes = plt.subplots(2, 2, figsize=(22, 16))

# 1.1 风险极差排名
y_pos = np.arange(len(variation_df))
colors_range = []
for val in variation_df['Risk_Range']:
    if val < LOW_VARIATION_THRESHOLD:
        colors_range.append('#27ae60')  # 绿色
    elif val < MODERATE_VARIATION_THRESHOLD:
        colors_range.append('#f39c12')  # 橙色
    else:
        colors_range.append('#e74c3c')  # 红色

axes[0, 0].barh(y_pos, variation_df['Risk_Range'], color=colors_range, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0, 0].set_title('各行业职位类型风险极差 (Max - Min)\\n绿色=一荣俱荣(<3%), 橙色=轻度分化(3-6%), 红色=显著分化(>6%)',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('风险极差 (百分点)', fontproperties=font, fontsize=13)
axes[0, 0].set_yticks(y_pos)
axes[0, 0].set_yticklabels(variation_df['Industry'], fontproperties=font, fontsize=12)
axes[0, 0].axvline(LOW_VARIATION_THRESHOLD, color='green', linestyle='--', linewidth=2, alpha=0.7, label='低变异阈值(3%)')
axes[0, 0].axvline(MODERATE_VARIATION_THRESHOLD, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='中等变异阈值(6%)')
for label in axes[0, 0].get_xticklabels():
    label.set_fontproperties(font)
axes[0, 0].grid(True, alpha=0.3, axis='x')
axes[0, 0].legend(prop=font, fontsize=11)
for i, val in enumerate(variation_df['Risk_Range']):
    axes[0, 0].text(val + 0.2, i, f'{val:.2f}%', va='center', fontproperties=font, fontsize=11, fontweight='bold')

# 1.2 变异系数(CV)排名
axes[0, 1].barh(y_pos, variation_df['Risk_CV'], color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0, 1].set_title('各行业风险变异系数 (CV = σ/μ)\\nCV越大,内部分化越严重',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('变异系数', fontproperties=font, fontsize=13)
axes[0, 1].set_yticks(y_pos)
axes[0, 1].set_yticklabels(variation_df['Industry'], fontproperties=font, fontsize=12)
for label in axes[0, 1].get_xticklabels():
    label.set_fontproperties(font)
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, val in enumerate(variation_df['Risk_CV']):
    axes[0, 1].text(val + 0.002, i, f'{val:.4f}', va='center', fontproperties=font, fontsize=10)

# 1.3 风险范围可视化 (Min-Max bars)
for i, row in variation_df.iterrows():
    industry = row['Industry']
    risk_min = row['Risk_Min']
    risk_max = row['Risk_Max']
    risk_range = row['Risk_Range']

    # 找到该行业在排序后的位置
    y_idx = variation_df.index.get_loc(i)

    # 绘制范围条
    axes[1, 0].plot([risk_min, risk_max], [y_idx, y_idx],
                    color=colors_range[y_idx], linewidth=8, alpha=0.7, solid_capstyle='round')
    # 标注最小值
    axes[1, 0].scatter(risk_min, y_idx, color='green', s=150, zorder=3, edgecolors='black', linewidth=1.5)
    # 标注最大值
    axes[1, 0].scatter(risk_max, y_idx, color='red', s=150, zorder=3, edgecolors='black', linewidth=1.5)

axes[1, 0].set_title('各行业职位类型风���分布范围\\n(绿点=最低风险职位, 红点=最高风险职位)',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('自动化风险 (%)', fontproperties=font, fontsize=13)
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels(variation_df['Industry'], fontproperties=font, fontsize=12)
for label in axes[1, 0].get_xticklabels():
    label.set_fontproperties(font)
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 1.4 关键洞察文本
axes[1, 1].axis('off')

# 统计不同模式的行业数量
unified_count = len(variation_df[variation_df['Risk_Range'] < LOW_VARIATION_THRESHOLD])
moderate_count = len(variation_df[(variation_df['Risk_Range'] >= LOW_VARIATION_THRESHOLD) &
                                   (variation_df['Risk_Range'] < MODERATE_VARIATION_THRESHOLD)])
divided_count = len(variation_df[variation_df['Risk_Range'] >= MODERATE_VARIATION_THRESHOLD])

most_unified = variation_df.iloc[0]['Industry'] if len(variation_df) > 0 else "N/A"
most_divided = variation_df.iloc[-1]['Industry'] if len(variation_df) > 0 else "N/A"
most_unified_range = variation_df.iloc[0]['Risk_Range'] if len(variation_df) > 0 else 0
most_divided_range = variation_df.iloc[-1]['Risk_Range'] if len(variation_df) > 0 else 0

insight_text = f"""
🔍 跨行业职位类型风险变异分析 - 核心洞察

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 行业内部模式分布:

   ✅ 一荣俱荣型 (<3%极差): {unified_count} 个行业
   ⚠️  轻度分化型 (3-6%极差): {moderate_count} 个行业
   ❌ 显著分化型 (>6%极差): {divided_count} 个行业

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 最统一行业: {most_unified}
   风险极差: {most_unified_range:.2f}%
   → 行业效应主导,职位类型影响微小

💥 最分化行业: {most_divided}
   风险极差: {most_divided_range:.2f}%
   → 职位类型选择至关重要!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ 核心结论:

1️⃣ IT的小变异(1.59%){'是' if most_unified == 'IT' else '不是'}个例
   {'→ IT确实是最统一的行业之一' if unified_count > 0 and 'IT' in variation_df[variation_df['Risk_Range'] < LOW_VARIATION_THRESHOLD]['Industry'].values else '→ 其他行业也表现出类似模式'}

2️⃣ {"大部分行业呈'一荣俱荣'模式" if unified_count >= len(variation_df)/2 else "行业间存在显著差异"}
   → {"行业标签比职位标签更重要" if unified_count >= len(variation_df)/2 else "既要选对行业,也要选对职位类型"}

3️⃣ 策略建议:
   • 统一型行业: 进入该行业即可,职位无需过度纠结
   • 分化型行业: 必须精挑细选职位类型!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 对个体的启示:

   在选择职业时:
   Step 1: 优先选对行业 (解释45%差异)
   Step 2: 检查该行业是统一型还是分化型
   Step 3: 如果是分化型,再精选职位类型 (+6-7%)
"""

axes[1, 1].text(0.05, 0.95, insight_text, transform=axes[1, 1].transAxes,
                fontproperties=font, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C1_outputs/ALL_industries_job_type_variation.png',
            dpi=300, bbox_inches='tight')
print("✓ ALL_industries_job_type_variation.png")
plt.close()

# 图2: 每个行业的详细职位类型风险分布 (多子图)
n_industries = len(industries)
n_cols = 3
n_rows = (n_industries + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
axes = axes.flatten() if n_industries > 1 else [axes]

for idx, industry in enumerate(sorted(industries)):
    industry_data = df[df['Industry'] == industry]

    job_type_stats = industry_data.groupby('Job_Type').agg({
        'Automation Risk (%)': ['mean', 'count']
    }).round(2)

    job_type_stats.columns = ['Risk_Mean', 'Count']
    job_type_stats = job_type_stats[job_type_stats['Count'] >= 30].sort_values('Risk_Mean')

    if len(job_type_stats) > 0:
        x_pos = np.arange(len(job_type_stats))
        risk_range = job_type_stats['Risk_Mean'].max() - job_type_stats['Risk_Mean'].min()

        # 根据变异程度着色
        if risk_range < LOW_VARIATION_THRESHOLD:
            bar_color = '#27ae60'
            pattern_label = "一荣俱荣"
        elif risk_range < MODERATE_VARIATION_THRESHOLD:
            bar_color = '#f39c12'
            pattern_label = "轻度分化"
        else:
            bar_color = '#e74c3c'
            pattern_label = "显著分化"

        axes[idx].bar(x_pos, job_type_stats['Risk_Mean'], color=bar_color,
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[idx].set_title(f'{industry}\\n极差={risk_range:.2f}% ({pattern_label})',
                           fontproperties=font, fontsize=13, fontweight='bold', pad=10)
        axes[idx].set_ylabel('风险 (%)', fontproperties=font, fontsize=11)
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(job_type_stats.index, fontproperties=font,
                                   fontsize=9, rotation=45, ha='right')
        for label in axes[idx].get_yticklabels():
            label.set_fontproperties(font)
        axes[idx].grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, val in enumerate(job_type_stats['Risk_Mean']):
            axes[idx].text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom',
                          fontproperties=font, fontsize=9)

# 隐藏多余的子图
for idx in range(len(industries), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C1_outputs/ALL_industries_detailed_job_types.png',
            dpi=300, bbox_inches='tight')
print("✓ ALL_industries_detailed_job_types.png")
plt.close()

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)
print("\n生成的文件:")
print("  1. ALL_industries_job_type_variation.png - 跨行业变异对比(四维)")
print("  2. ALL_industries_detailed_job_types.png - 每个行业详细分布")
print("  3. industry_internal_variation_summary.csv - 变异汇总数据")
print("\n核心发现:")
print(f"  • 一荣俱荣型: {unified_count}/{len(variation_df)} 个行业")
print(f"  • 轻度分化型: {moderate_count}/{len(variation_df)} 个行业")
print(f"  • 显著分化型: {divided_count}/{len(variation_df)} 个行业")
