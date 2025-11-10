import pandas as pd
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')

print('='*80)
print('关键字段数据生成机制检查')
print('='*80)

# 1. Automation Risk
print('\n【Automation Risk (%)】')
stats = df['Automation Risk (%)'].describe()
print(stats)
actual_std = df['Automation Risk (%)'].std()
theoretical_std = 100/np.sqrt(12)
print(f'标准差: {actual_std:.2f}')
print(f'理论均匀分布[0,100]标准差: {theoretical_std:.2f}')
if abs(actual_std - theoretical_std) < 5:
    print('判断: 可能是均匀分布')
else:
    print('判断: 不是均匀分布,可能是真实建模')

# 2. Gender Diversity
print('\n【Gender Diversity (%)】')
stats = df['Gender Diversity (%)'].describe()
print(stats)
actual_std = df['Gender Diversity (%)'].std()
theoretical_std = 100/np.sqrt(12)
print(f'标准差: {actual_std:.2f}')
print(f'理论均匀分布[0,100]标准差: {theoretical_std:.2f}')
if abs(actual_std - theoretical_std) < 5:
    print('判断: 可能是均匀分布')
else:
    print('判断: 不是均匀分布,可能是真实建模')

# 3. Median Salary
print('\n【Median Salary (USD)】')
stats = df['Median Salary (USD)'].describe()
print(stats)
salary_std = df['Median Salary (USD)'].std()
salary_range = df['Median Salary (USD)'].max() - df['Median Salary (USD)'].min()
theoretical_std = salary_range / np.sqrt(12)
print(f'标准差: ${salary_std:,.2f}')
print(f'如果在范围内均匀分布,理论标准差: ${theoretical_std:,.2f}')
if abs(salary_std - theoretical_std) / theoretical_std < 0.1:
    print('判断: 可能是均匀分布')
else:
    print('判断: 不是均匀分布,可能是真实建模')

# 检查各行业Gender Diversity
print('\n' + '='*80)
print('各行业Gender Diversity平均值:')
print('='*80)
for ind in sorted(df['Industry'].unique()):
    mean = df[df['Industry'] == ind]['Gender Diversity (%)'].mean()
    print(f'{ind:15s}: {mean:.2f}%')

# 检查各行业Automation Risk
print('\n' + '='*80)
print('各行业Automation Risk平均值:')
print('='*80)
for ind in sorted(df['Industry'].unique()):
    mean = df[df['Industry'] == ind]['Automation Risk (%)'].mean()
    std = df[df['Industry'] == ind]['Automation Risk (%)'].std()
    print(f'{ind:15s}: {mean:.2f}% (std={std:.2f})')
