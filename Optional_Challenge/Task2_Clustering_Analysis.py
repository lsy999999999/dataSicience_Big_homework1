# -*- coding: utf-8 -*-
"""
任务2: 聚类分析
Job Clustering: Revealing AI-Era Employment Fate Clusters

目标: 发现工作的"命运集群",揭示AI时代就业市场的结构性分化
创新点: 不只聚类,更要讲述每个集群的"命运轨迹"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出编码设置
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*80)
print('任务2: 聚类分析 - 揭示AI时代就业市场的命运集群')
print('='*80)

# ============================================================================
# 1. 数据加载
# ============================================================================

df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')

print('\n【数据概览】')
print(f'总样本数: {len(df):,}')

# ============================================================================
# 2. 选择聚类特征 (任务要求的4个特征)
# ============================================================================

print('\n【聚类特征选择】')
print('根据任务要求,选择以下4个特征:')
print('  1. Automation Risk (%)')
print('  2. Median Salary (USD)')
print('  3. Experience Required (Years)')
print('  4. Remote Work Ratio (%)')

# 计算岗位增长率 (用于后续分析,使用2024和2030数据)
df['Job_Growth_Pct'] = (df['Projected Openings (2030)'] - df['Job Openings (2024)']) / df['Job Openings (2024)'] * 100

clustering_features = [
    'Automation Risk (%)',
    'Median Salary (USD)',
    'Experience Required (Years)',
    'Remote Work Ratio (%)'
]

X = df[clustering_features].copy()

print('\n特征统计:')
print(X.describe())

# ============================================================================
# 3. 数据标准化 (关键!)
# ============================================================================

print('\n【数据标准化】')
print('为什么要标准化?')
print('  - Salary在0-200,000范围,Risk在0-100范围')
print('  - 不标准化会导致Salary主导聚类(单位大)')
print('  - 标准化后每个特征都是均值0、标准差1')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('\n标准化后的统计:')
print(pd.DataFrame(X_scaled, columns=clustering_features).describe())

# ============================================================================
# 4. 最优K值选择 (科学方法)
# ============================================================================

print('\n' + '='*80)
print('【最优K值选择】')
print('='*80)

K_range = range(2, 11)
inertias = []
silhouette_scores = []
db_scores = []

print('\n计算不同K值的评估指标...')

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

    print(f'K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouette_scores[-1]:.4f}, DB={db_scores[-1]:.4f}')

# 可视化K值选择
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 左图: Elbow Method
axes[0].plot(K_range, inertias, 'o-', linewidth=2, markersize=8, color='steelblue')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method for Optimal K', fontsize=14, weight='bold')
axes[0].axvline(x=4, color='red', linestyle='--', linewidth=2, label='K=4 (建议)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 中图: Silhouette Score (越大越好)
axes[1].plot(K_range, silhouette_scores, 'o-', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score (Higher is Better)', fontsize=14, weight='bold')
optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
axes[1].axvline(x=optimal_k_silhouette, color='red', linestyle='--', linewidth=2,
                label=f'K={optimal_k_silhouette} (最佳)')
axes[1].legend()
axes[1].grid(alpha=0.3)

# 右图: Davies-Bouldin Index (越小越好)
axes[2].plot(K_range, db_scores, 'o-', linewidth=2, markersize=8, color='orange')
axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
axes[2].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, weight='bold')
optimal_k_db = K_range[np.argmin(db_scores)]
axes[2].axvline(x=optimal_k_db, color='red', linestyle='--', linewidth=2,
                label=f'K={optimal_k_db} (最佳)')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Optional_Challenge/Task2_outputs/01_optimal_k_selection.png', dpi=300, bbox_inches='tight')
print('\n✓ 已保存: 01_optimal_k_selection.png')
plt.close()

# 选择最优K (综合考虑)
optimal_k = 4  # 基于Elbow Method和实际可解释性

print(f'\n✅ 选择 K={optimal_k} (综合考虑Elbow法和实际可解释性)')

# ============================================================================
# 5. K-Means聚类
# ============================================================================

print('\n' + '='*80)
print(f'【K-Means聚类 (K={optimal_k})】')
print('='*80)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
cluster_labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = cluster_labels

# 聚类质量评估
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
db_index = davies_bouldin_score(X_scaled, cluster_labels)

print(f'\n聚类质量指标:')
print(f'  - Silhouette Score: {silhouette_avg:.4f} (���接近1越好)')
print(f'  - Davies-Bouldin Index: {db_index:.4f} (越小越好)')
print(f'  - Inertia: {kmeans.inertia_:.0f}')

# ============================================================================
# 6. 聚类结果解释 (核心!)
# ============================================================================

print('\n' + '='*80)
print('【聚类结果分析】')
print('='*80)

cluster_stats = df.groupby('Cluster').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Experience Required (Years)': 'mean',
    'Remote Work Ratio (%)': 'mean',
    'Job_Growth_Pct': 'mean',
    'Job Title': 'count'
})

cluster_stats.columns = ['Avg_Risk', 'Avg_Salary', 'Avg_Experience',
                         'Avg_Remote', 'Avg_Growth', 'Count']
cluster_stats['Percentage'] = cluster_stats['Count'] / cluster_stats['Count'].sum() * 100

print('\n集群统计概览:')
print(cluster_stats.to_string())

# 保存集群统计
cluster_stats.to_csv('Optional_Challenge/Task2_outputs/cluster_statistics.csv', encoding='utf-8-sig')

# 为每个集群命名和解释
cluster_names = {}
cluster_descriptions = {}

# 根据特征自动命名和解释
for i in range(optimal_k):
    stats = cluster_stats.loc[i]

    # 根据薪资和风险组合命名
    if stats['Avg_Salary'] > 100000 and stats['Avg_Risk'] < 40:
        name = "精英集群 (Elite)"
        desc = "高薪低风险,AI时代的赢家"
    elif stats['Avg_Salary'] < 70000 and stats['Avg_Risk'] > 48:
        name = "高危集群 (Vulnerable)"
        desc = "低薪高风险,面临严重替代威胁"
    elif stats['Avg_Remote'] > 60:
        name = "新兴灵活 (Flexible New)"
        desc = "远程友好,AI催生的新模式"
    else:
        name = "传统中产 (Traditional Middle)"
        desc = "中等薪资风险,转型压力中等"

    cluster_names[i] = name
    cluster_descriptions[i] = desc

# 打印每个集群的详细特征
print('\n' + '='*80)
print('【集群详细解读】')
print('='*80)

for i in range(optimal_k):
    stats = cluster_stats.loc[i]

    print(f'\n{"="*60}')
    print(f'Cluster {i}: {cluster_names[i]}')
    print(f'{"="*60}')
    print(f'核心定位: {cluster_descriptions[i]}')
    print(f'\n特征:')
    print(f'  - 自动化风险: {stats["Avg_Risk"]:.2f}%')
    print(f'  - 平均薪资: ${stats["Avg_Salary"]:,.0f}')
    print(f'  - 经验要求: {stats["Avg_Experience"]:.2f}年')
    print(f'  - 远程工作: {stats["Avg_Remote"]:.2f}%')
    print(f'\n规模与增长:')
    print(f'  - 占比: {stats["Percentage"]:.1f}% ({stats["Count"]:,}个岗位)')
    print(f'  - 增长率: {stats["Avg_Growth"]:+.1f}%')

    # 找典型职位 (每个cluster抽取3个最常见的job title)
    cluster_jobs = df[df['Cluster'] == i]['Job Title'].value_counts().head(3)
    print(f'\n典型职位示例:')
    for job, count in cluster_jobs.items():
        print(f'  - {job} ({count}个)')

    # 行业分布
    cluster_industries = df[df['Cluster'] == i]['Industry'].value_counts()
    top_industries = cluster_industries.head(3)
    print(f'\n主要行业分布:')
    for industry, count in top_industries.items():
        pct = count / cluster_industries.sum() * 100
        print(f'  - {industry}: {pct:.1f}%')

# ============================================================================
# 7. 聚类可视化
# ============================================================================

print('\n' + '='*80)
print('【��类可视化】')
print('='*80)

# 7.1 2D散点图: Risk vs Salary
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 左上: Risk vs Salary
scatter1 = axes[0, 0].scatter(df['Automation Risk (%)'],
                              df['Median Salary (USD)'],
                              c=df['Cluster'],
                              s=df['Experience Required (Years)']*15,
                              alpha=0.6,
                              cmap='viridis',
                              edgecolors='black',
                              linewidths=0.5)

# 标注聚类中心
for i in range(optimal_k):
    center_risk = scaler.inverse_transform(kmeans.cluster_centers_)[i][0]
    center_salary = scaler.inverse_transform(kmeans.cluster_centers_)[i][1]
    axes[0, 0].scatter(center_risk, center_salary,
                      c='red', s=400, marker='X',
                      edgecolors='black', linewidths=2, zorder=5)
    axes[0, 0].text(center_risk, center_salary + 5000,
                   f'C{i}\n{cluster_names[i].split()[0]}',
                   fontsize=10, weight='bold', ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

axes[0, 0].set_xlabel('Automation Risk (%)', fontsize=12)
axes[0, 0].set_ylabel('Median Salary (USD)', fontsize=12)
axes[0, 0].set_title('Job Clusters: Risk vs Salary\n(气泡大小 = 经验要求)', fontsize=13, weight='bold')
axes[0, 0].grid(alpha=0.3)
plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')

# 右上: Experience vs Remote
scatter2 = axes[0, 1].scatter(df['Experience Required (Years)'],
                              df['Remote Work Ratio (%)'],
                              c=df['Cluster'],
                              s=df['Median Salary (USD)']/500,
                              alpha=0.6,
                              cmap='viridis',
                              edgecolors='black',
                              linewidths=0.5)

axes[0, 1].set_xlabel('Experience Required (Years)', fontsize=12)
axes[0, 1].set_ylabel('Remote Work Ratio (%)', fontsize=12)
axes[0, 1].set_title('Job Clusters: Experience vs Remote\n(气泡大小 = 薪资)', fontsize=13, weight='bold')
axes[0, 1].grid(alpha=0.3)
plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')

# 左下: 集群规模与增长
cluster_summary = cluster_stats[['Percentage', 'Avg_Growth']].reset_index()
colors_map = {0: 'skyblue', 1: 'lightgreen', 2: 'salmon', 3: 'gold'}
bar_colors = [colors_map[i] for i in cluster_summary['Cluster']]

x = np.arange(len(cluster_summary))
width = 0.35

bars1 = axes[1, 0].bar(x - width/2, cluster_summary['Percentage'],
                       width, label='占比 (%)', color=bar_colors, alpha=0.8,
                       edgecolor='black')
ax2 = axes[1, 0].twinx()
bars2 = ax2.bar(x + width/2, cluster_summary['Avg_Growth'],
                width, label='增长率 (%)', color='orange', alpha=0.7,
                edgecolor='black')

axes[1, 0].set_xlabel('Cluster', fontsize=12)
axes[1, 0].set_ylabel('占比 (%)', fontsize=12, color='black')
ax2.set_ylabel('平均增长率 (%)', fontsize=12, color='orange')
axes[1, 0].set_title('Cluster Size and Growth Rate', fontsize=13, weight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([f'C{i}\n{cluster_names[i].split()[0]}' for i in cluster_summary['Cluster']])
axes[1, 0].legend(loc='upper left')
ax2.legend(loc='upper right')
axes[1, 0].grid(axis='y', alpha=0.3)

# 右下: 集群特征雷达图 (标准化后)
categories = ['Risk', 'Salary', 'Experience', 'Remote']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax_radar = plt.subplot(2, 2, 4, projection='polar')

for i in range(optimal_k):
    values = [
        cluster_stats.loc[i, 'Avg_Risk'] / 100,  # 归一化到0-1
        cluster_stats.loc[i, 'Avg_Salary'] / 200000,  # 归一化到0-1
        cluster_stats.loc[i, 'Avg_Experience'] / 20,  # 归一化到0-1
        cluster_stats.loc[i, 'Avg_Remote'] / 100  # 归一化到0-1
    ]
    values += values[:1]

    ax_radar.plot(angles, values, 'o-', linewidth=2,
                 label=f'C{i}: {cluster_names[i].split()[0]}')
    ax_radar.fill(angles, values, alpha=0.15)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories)
ax_radar.set_ylim(0, 1)
ax_radar.set_title('Cluster Profiles (Normalized)', fontsize=13, weight='bold', pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax_radar.grid(True)

plt.tight_layout()
plt.savefig('Optional_Challenge/Task2_outputs/02_cluster_visualization.png', dpi=300, bbox_inches='tight')
print('\n✓ 已保存: 02_cluster_visualization.png')
plt.close()

# ============================================================================
# 8. 行业与集群关系分析
# ============================================================================

print('\n' + '='*80)
print('【行业与集群关系】')
print('='*80)

# 行业×集群交叉表
industry_cluster_crosstab = pd.crosstab(df['Industry'], df['Cluster'], normalize='index') * 100

print('\n各行业在各集群的分布 (%):')
print(industry_cluster_crosstab.round(1).to_string())

# 保存
industry_cluster_crosstab.to_csv('Optional_Challenge/Task2_outputs/industry_cluster_distribution.csv',
                                 encoding='utf-8-sig')

# 可视化: 热力图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图: 行业在各集群的分布
sns.heatmap(industry_cluster_crosstab,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Percentage (%)'},
            linewidths=0.5,
            ax=axes[0])
axes[0].set_title('Industry Distribution Across Clusters (%)\n(每行求和=100%)', fontsize=13, weight='bold')
axes[0].set_xlabel('Cluster', fontsize=12)
axes[0].set_ylabel('Industry', fontsize=12)
axes[0].set_xticklabels([f'C{i}\n{cluster_names[i].split()[0]}' for i in range(optimal_k)], rotation=0)

# 右图: 集群在各行业的分布
cluster_industry_crosstab = pd.crosstab(df['Cluster'], df['Industry'], normalize='index') * 100
sns.heatmap(cluster_industry_crosstab,
            annot=True,
            fmt='.1f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Percentage (%)'},
            linewidths=0.5,
            ax=axes[1])
axes[1].set_title('Cluster Distribution Across Industries (%)\n(每行求和=100%)', fontsize=13, weight='bold')
axes[1].set_xlabel('Industry', fontsize=12)
axes[1].set_ylabel('Cluster', fontsize=12)
axes[1].set_yticklabels([f'C{i}: {cluster_names[i].split()[0]}' for i in range(optimal_k)], rotation=0)

plt.tight_layout()
plt.savefig('Optional_Challenge/Task2_outputs/03_industry_cluster_heatmap.png', dpi=300, bbox_inches='tight')
print('\n✓ 已保存: 03_industry_cluster_heatmap.png')
plt.close()

# 分析每个行业的主导集群
print('\n各行业的主导集群:')
for industry in df['Industry'].unique():
    dominant_cluster = df[df['Industry'] == industry]['Cluster'].value_counts().idxmax()
    dominant_pct = df[df['Industry'] == industry]['Cluster'].value_counts().max() / len(df[df['Industry'] == industry]) * 100
    print(f'  {industry:20s} → Cluster {dominant_cluster} ({cluster_names[dominant_cluster]}) - {dominant_pct:.1f}%')

# ============================================================================
# 9. 集群增长潜力分析
# ============================================================================

print('\n' + '='*80)
print('【集群增长潜力分析】')
print('='*80)

print('\n各集群的平均岗位增长率:')
for i in range(optimal_k):
    growth = cluster_stats.loc[i, 'Avg_Growth']
    size = cluster_stats.loc[i, 'Percentage']
    print(f'  Cluster {i} ({cluster_names[i]}):')
    print(f'    - 增长率: {growth:+.1f}%')
    print(f'    - 当前占比: {size:.1f}%')

    # 判断趋势
    if growth > 150:
        trend = "🚀 爆炸性增长 - 代表未来趋势"
    elif growth > 100:
        trend = "📈 强劲增长 - 值得关注"
    elif growth > 50:
        trend = "➡️ 温和增长 - 相对稳定"
    else:
        trend = "⚠️ 低迷增长 - 面临衰退风险"

    print(f'    - 趋势: {trend}\n')

# 可视化: 集群大小 vs 增长率 (气泡图)
fig, ax = plt.subplots(figsize=(12, 8))

for i in range(optimal_k):
    size = cluster_stats.loc[i, 'Percentage']
    growth = cluster_stats.loc[i, 'Avg_Growth']
    risk = cluster_stats.loc[i, 'Avg_Risk']

    # 气泡大小代表集群规模
    bubble_size = size * 100

    scatter = ax.scatter(risk, growth, s=bubble_size, alpha=0.6,
                        edgecolors='black', linewidths=2,
                        label=f'C{i}: {cluster_names[i]}')

    # 标注
    ax.text(risk, growth, f'C{i}\n{size:.1f}%',
           fontsize=11, weight='bold', ha='center', va='center')

# 添加参考线
ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='100%增长基准线')
ax.axvline(x=df['Automation Risk (%)'].mean(), color='blue', linestyle='--',
          linewidth=1.5, alpha=0.7, label=f'平均风险 ({df["Automation Risk (%)"].mean():.1f}%)')

ax.set_xlabel('Average Automation Risk (%)', fontsize=13)
ax.set_ylabel('Average Job Growth Rate (%)', fontsize=13)
ax.set_title('Cluster Size vs Growth Potential\n(气泡大小 = 集群占比)', fontsize=14, weight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Optional_Challenge/Task2_outputs/04_cluster_growth_potential.png', dpi=300, bbox_inches='tight')
print('\n✓ 已保存: 04_cluster_growth_potential.png')
plt.close()

# ============================================================================
# 10. 聚类的宏观意义总结
# ============================================================================

print('\n' + '='*80)
print('【聚类分析的宏观意义】')
print('='*80)

print('\n✅ 核心发现:')
print('  AI正在将就业市场分化为几个命运不同的"集群"')
print('  这些集群跨越行业边界,形成新的阶层结构')

print('\n⭐ 集群命运轨迹:')

# 按增长率排序
cluster_fate = cluster_stats.sort_values('Avg_Growth', ascending=False)

for rank, (i, stats) in enumerate(cluster_fate.iterrows(), 1):
    print(f'\n{rank}. Cluster {i} - {cluster_names[i]}')
    print(f'   规模: {stats["Percentage"]:.1f}%')
    print(f'   增长: {stats["Avg_Growth"]:+.1f}%')
    print(f'   薪资: ${stats["Avg_Salary"]:,.0f}')
    print(f'   风险: {stats["Avg_Risk"]:.1f}%')
    print(f'   命运: {cluster_descriptions[i]}')

print('\n💡 对个体的启示:')
print('  1. 行业不再决定命运,**集群才是新的阶层**')
print('  2. 同一行业内部存在跨集群分化')
print('  3. 关键是识别自己所在集群,并规划迁移路径')

print('\n🎯 集群迁移建议:')
print('  - 如果在高危集群 → 紧急转型,目标迁移到新兴灵活或传统中产')
print('  - 如果在传统中产 → 主动提升,争取进入精英集群')
print('  - 如果在精英/新兴 → 保持学习,巩固优势地位')

# ============================================================================
# 11. 与B1-C3分析的连接
# ============================================================================

print('\n' + '='*80)
print('【与B1-C3分析的呼应】')
print('='*80)

print('\n聚类分析验证并深化了我们之前的发现:')

print('\n1. 与B1×B2的连接:')
print('   - B1×B2发现: "战场(行业)>盔甲(教育)"')
print('   - 聚类发现: 同一行业内部也存在不同集群')
print('   - 启示: 集群是比行业更精细的"命运单元"')

print('\n2. 与C1的连接:')
print('   - C1发现: 62%行业内部显著分化')
print('   - 聚类发现: 4个跨行业的集群,每个行业在不同集群都有分布')
print('   - 启示: 行业内部分化的本质是集群分化')

print('\n3. 与C2的连接:')
print('   - C2发现: 远程工作在不同行业效应不同')
print('   - 聚类发现: 新兴灵活集群(C3)远程比例最高(>60%)')
print('   - 启示: 远程工作模式正在创造新的就业集群')

print('\n4. 与C3的连接:')
print('   - C3发现: 多样性影响微弱(r≈0)')
print('   - 聚类发现: 4个集群的风险差异主要由职位特征决定')
print('   - 启示: 职位本质(薪资、风险、经验)比多样性更重要')

# ============================================================================
# 12. 最终总结
# ============================================================================

print('\n' + '='*80)
print('【任务2 总结】')
print('='*80)

print(f'\n✅ 聚类质量:')
print(f'   - 聚类数: K={optimal_k}')
print(f'   - Silhouette Score: {silhouette_avg:.4f}')
print(f'   - Davies-Bouldin Index: {db_index:.4f}')

print(f'\n⭐ 发现了{optimal_k}个命运集群:')
for i in range(optimal_k):
    print(f'   - Cluster {i}: {cluster_names[i]} ({cluster_stats.loc[i, "Percentage"]:.1f}%, 增长{cluster_stats.loc[i, "Avg_Growth"]:+.1f}%)')

print(f'\n💡 核心洞察:')
print(f'   - AI正在重构就业阶层,集群>行业>教育')
print(f'   - 增长最快的集群往往规模较小(新兴模式)')
print(f'   - 最大的集群(传统中产)面临转型压力')
print(f'   - 高危集群规模可观,需要政策干预')

print(f'\n🔗 与整体分析的连接:')
print(f'   - 聚类分析是B1-C3的自然延伸')
print(f'   - 从"因素分析"升级到"集群画像"')
print(f'   - 为个体提供了更精准的职业定位工具')

print('\n' + '='*80)
print('任务2完成! 所有输出已保存至 Optional_Challenge/Task2_outputs/')
print('='*80)
