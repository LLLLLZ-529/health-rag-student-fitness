# M3 完整推荐pipeline：9类学生案例

流程：学生健康分类 → RAG检索知识库 → LinUCB选策略 → 生成推荐理由

**策略选择准确率：9/9 = 100.0%**

## 低水平-退化型（class 0）

- 该类学生数：5289
- **推荐策略：运动+膳食综合**
- 先验最优策略：运动+膳食综合
- 策略匹配：✅ 正确

### 推荐理由

您的体质处于较低水平且呈下降趋势，需要运动和膳食的综合干预。建议在专业指导下制定个性化训练计划，同时调整饮食结构。

### 体测概况

- bmi: 29.42
- vital_capacity: 1690.0
- sprint_50m: 10.8
- standing_long_jump: 168.0
- sit_and_reach: 17.9
- endurance_run_sec: 303.0
- strength: 29.0

### RAG检索结果

## 低水平-平稳型（class 1）

- 该类学生数：6160
- **推荐策略：运动干预为主**
- 先验最优策略：运动干预为主
- 策略匹配：✅ 正确

### 推荐理由

您的体质处于较低水平但保持稳定，运动干预是提升的关键。建议从有氧运动入手，每周3-4次，逐步提升体能。

### 体测概况

- bmi: 16.21
- vital_capacity: 2212.0
- sprint_50m: 8.2
- standing_long_jump: 172.0
- sit_and_reach: 12.9
- endurance_run_sec: 333.0
- strength: 19.0

### RAG检索结果

## 低水平-改善型（class 2）

- 该类学生数：2251
- **推荐策略：维持现状+监测**
- 先验最优策略：维持现状+监测
- 策略匹配：✅ 正确

### 推荐理由

您的体质正在改善，建议维持当前的运动和饮食习惯，定期监测体测指标。

### 体测概况

- bmi: 22.45
- vital_capacity: 2372.0
- sprint_50m: 8.6
- standing_long_jump: 162.0
- sit_and_reach: 8.1
- endurance_run_sec: 376.0
- strength: 38.0

### RAG检索结果

## 中水平-退化型（class 3）

- 该类学生数：3816
- **推荐策略：运动+膳食综合**
- 先验最优策略：运动+膳食综合
- 策略匹配：✅ 正确

### 推荐理由

您的体质中等但呈下降趋势，需要综合干预来逆转。建议制定系统的训练计划，同时调整饮食和作息。

### 体测概况

- bmi: 20.06
- vital_capacity: 2502.0
- sprint_50m: 9.0
- standing_long_jump: 158.0
- sit_and_reach: 12.1
- endurance_run_sec: 256.0
- strength: 35.0

### RAG检索结果

## 中水平-平稳型（class 4）

- 该类学生数：10595
- **推荐策略：运动干预为主**
- 先验最优策略：运动干预为主
- 策略匹配：✅ 正确

### 推荐理由

您的体质中等且稳定，运动干预可以帮助突破平台期。建议增加训练强度或尝试新的运动方式。

### 体测概况

- bmi: 21.41
- vital_capacity: 3351.0
- sprint_50m: 10.0
- standing_long_jump: 148.0
- sit_and_reach: 18.3
- endurance_run_sec: 377.0
- strength: 35.0

### RAG检索结果

## 中水平-改善型（class 5）

- 该类学生数：6509
- **推荐策略：维持现状+监测**
- 先验最优策略：维持现状+监测
- 策略匹配：✅ 正确

### 推荐理由

您的体质正在改善，建议维持当前的健康生活方式，定期监测体测变化。

### 体测概况

- bmi: 18.76
- vital_capacity: 2852.0
- sprint_50m: 9.4
- standing_long_jump: 191.0
- sit_and_reach: 20.2
- endurance_run_sec: 300.0
- strength: 19.0

### RAG检索结果

## 高水平-退化型（class 6）

- 该类学生数：93
- **推荐策略：膳食干预为主**
- 先验最优策略：膳食干预为主
- 策略匹配：✅ 正确

### 推荐理由

您的体质优秀但呈下降趋势，膳食调整可能是关键。建议关注营养摄入是否充足，特别是蛋白质和微量元素。

### 体测概况

- bmi: 22.31
- vital_capacity: 3390.0
- sprint_50m: 8.5
- standing_long_jump: 202.0
- sit_and_reach: 24.5
- endurance_run_sec: 263.0
- strength: 46.0

### RAG检索结果

## 高水平-平稳型（class 7）

- 该类学生数：626
- **推荐策略：维持现状+监测**
- 先验最优策略：维持现状+监测
- 策略匹配：✅ 正确

### 推荐理由

您的体质优秀且稳定，建议维持当前的健康生活方式，这是最佳状态。

### 体测概况

- bmi: 22.28
- vital_capacity: 3192.0
- sprint_50m: 8.7
- standing_long_jump: 190.0
- sit_and_reach: 26.0
- endurance_run_sec: 243.0
- strength: 38.0

### RAG检索结果

## 高水平-改善型（class 8）

- 该类学生数：720
- **推荐策略：维持现状+监测**
- 先验最优策略：维持现状+监测
- 策略匹配：✅ 正确

### 推荐理由

您的体质优秀且持续进步，建议维持当前状态，这是非常理想的健康趋势。

### 体测概况

- bmi: 19.45
- vital_capacity: 2957.0
- sprint_50m: 8.3
- standing_long_jump: 193.0
- sit_and_reach: 24.4
- endurance_run_sec: 239.0
- strength: 40.0

### RAG检索结果

