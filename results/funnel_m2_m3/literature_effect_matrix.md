# M3 干预效果矩阵：文献依据版

## 文献效应量来源

| 来源 | 指标 | 效应量 | 换算 z-score |
|------|------|--------|-------------|
| Chase & Conn (2015) meta-analysis, 11,458 subjects | VO2max (心肺) | SMD=0.48 (一般), 1.04 (耐力+抗阻综合) | 0.48 / 1.04 |
| Wang et al. (2024) NMA, 28 studies | BMI (有氧运动) | WMD=-0.9 kg/m² | -0.9/3.5 ≈ -0.26 |
| Chen et al. (2025) NMA, 93 RCTs | VO2max (HIIT) | SMD=4.85 | 4.85 (极端值，取保守1.0) |
| Lin et al. (2015) JAH meta | VO2max | SMD≈0.5-0.8 | 0.65 |
| Barrantes-Brais et al. (2013) | 大学生运动干预 | ES=0.57 | 0.57 |
| Shu et al. (2024) PLOS ONE | 有氧运动对中国大学生 | 16周效果最大 | 取16周=1.0单位 |

## 各指标效应量估计（12周干预，z-score 单位）

方向：BMI↓, VC↑, 50m↓, 跳远↑, 体前屈↑, 耐力↓, 力量↑

### 运动干预（有氧运动为主）
- BMI: -0.26 (Wang 2024, AE -0.9 kg/m²)
- VC/心肺: +0.65 (Lin 2015, SMD 0.65)
- 50m: -0.15 (间接效应，小)
- 跳远: +0.30 (心肺改善带动)
- 体前屈: +0.10 (有氧运动对柔韧小)
- 耐力跑: -0.50 (有氧运动直接效果，SMD≈0.5)
- 力量: +0.15 (有氧运动对力量小)

### 膳食干预
- BMI: -0.35 (饮食+运动比单纯运动减重更多)
- VC: +0.05 (间接)
- 50m: -0.02
- 跳远: +0.05
- 体前屈: +0.05
- 耐力: -0.05
- 力量: +0.02

### 运动+膳食综合
- BMI: -0.45 (两者叠加)
- VC: +0.80 (Chase 2015: 综合 SMD=1.04, 取保守0.8)
- 50m: -0.20
- 跳远: +0.40
- 体前屈: +0.15
- 耐力: -0.65
- 力量: +0.35

### 维持现状
- 全 0（自然状态）

## 更新后的 EFFECT_MATRIX

```python
# 顺序: bmi↓ vc↑ sprint↓ jump↑ reach↑ endurance↓ strength↑
EFFECT_MATRIX = np.array([
    [-0.26, 0.65, -0.15, 0.30, 0.10, -0.50, 0.15],  # 运动
    [-0.35, 0.05, -0.02, 0.05, 0.05, -0.05, 0.02],  # 膳食
    [-0.45, 0.80, -0.20, 0.40, 0.15, -0.65, 0.35],  # 综合
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],     # 维持
])
```

## 引用
1. Chase JD, Conn VS. Meta-analysis of fitness outcomes from motivational PA interventions. Nurs Res 2015.
2. Wang H et al. Comparative efficacy of exercise training modes on metabolic health. Front Endocrinol 2024.
3. Chen Z et al. Comparative effectiveness of exercise interventions on CRF. J Sports Sci 2025.
4. Lin X et al. Effects of exercise training on cardiorespiratory fitness. J Am Heart Assoc 2015.
5. Barrantes-Brais K et al. Exercise interventions on college students' well-being: meta-analysis. JSHR 2013.
6. Shu J et al. Effects of aerobic exercise on body self-esteem among Chinese college students: meta-analysis. PLoS ONE 2024.
