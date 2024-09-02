# StockJump

本文为复现https://arxiv.org/abs/2404.16467的代码，如果对您有帮助请star~ ;D

计算jump_score的规则为：

#### Jt = rt/(σt*ft)

其中rt为midprice对数收益率，

#### rt = log( (1/2 * (highprice(t) + lowprice(t))) / (1/2 * (highprice(t-1) + lowprice(t-1))) )，

![image](https://github.com/user-attachments/assets/71ee00c0-744b-4d8b-985a-b1cf459b0268)

![image](https://github.com/user-attachments/assets/eb5f501a-f2f0-4216-af2f-8d85c90af899)

![image](https://github.com/user-attachments/assets/9803643b-d847-492a-b391-640a982eae6d)

![image](https://github.com/user-attachments/assets/2afca299-cd29-4f78-93ef-4f34ccbb6423)

![image](https://github.com/user-attachments/assets/7d0e4c78-eb9f-4a7a-971b-a81bb3aa64d9)

![image](https://github.com/user-attachments/assets/e6e5b0f5-0928-482b-b4fc-6a3ce2dff954)

针对每支股票，在每条数据的时间节点向前滚动K条数据，计算上述参数，得到每个时间节点的jump_score

满足：

![image](https://github.com/user-attachments/assets/3172d03f-a238-4794-bc69-538002be9e4f)

![image](https://github.com/user-attachments/assets/dc0cc69e-207d-49f9-bee3-65eda8458857)

的则被识别为价格跳跃，按其jump_score的正负分为正跳和负跳。

分别提取出跳跃前后各59条（可变）数据，即可计算出每个跳跃的D1,D2,D3分数，作为区分其为何种跳跃的因子值。

其中，每个跳跃的D1分数计算方法如下：

#### 每个跳跃前后共119条jump_score数据组成的jump_array(,119)与42个不同尺度的滤波器分别卷积得到arr1(, 42)，

#### 在该跳跃发生的时间节点向前回看W=300条跳跃数据得到arr2(300,42)，对arr2做核PCA得到主成分权重w(,42)，

#### 提取出w(,42)中形如IMfj1(t) * j2(t)共15个分量分别的权重w1(,15)，再将每个分量分别乘以权重并求和得到每个跳跃的D1分数；

#### 而D2分数则为其中IMf0(t)分量对应的值，D3分数为其中REf0(t)分量对应的值。

准确起见，将原文内容陈列如下：

        The first PCA direction (called D1 henceforth) is a linear combination of the 15 coefficients Im Wj2|Wj1 x|(0) in Eq. (7),
        
        which characterizes time-asymmetry of the volatility profile at multiple scales 2j2, confirming previous analysis that postulated this asymmetry to be relevant. 
        
        Such a linear combination allows one to embed each jump time-series into a one dimensional space, which quantifies the reflexive nature of each jump. 
        
        In fact, Fig. 4 and Appendix C 2 display average profiles |x(t)| along the “reflexive direction” D1.
        
        One can visually verify that such a representation discriminates jumps according to the asymmetry of their profiles as measured by Ajump (Eq. (2)): 
        
        the D1 direction continuously separates asymmetric jumps with dominant activity before the shock from asymmetric jumps with dominant activity after the shock; see Figs. 4, 18 and 21a.
        
        We observed that coefficients Im Wj1 x(0) (7) for fine scales, i.e. small j1, are consistently chosen by the leading PCA directions.
        
        They amount to multiplying the jump-aligned time-series x(t) by the imaginary filter Im ψ1(t) (ψMR) and averaging over t. 
        
        Such coefficients capture the asymmetry of the return profile shortly before and shortly after the jump, and define what we will call below direction D2.
        
        In the previous section, we have defined a filter ψMR that detects mean-reversion, but is by construction orthogonal to trends, 
        
        i.e. post-jump returns continuing in the same direction as pre-jump returns. 
        
        This feature can be naturally captured by the trend filter ψTR shown in Fig. 5, which is orthogonal to the mean-reversion filter ψMR. 
        
        This filter is then applied to the jump-aligned profile x(t) to get the following trend score De3(x) := x ⋆ ψTR(0). 

理论中，D1,D2,D3的值能够区分的跳跃形态如下：

![image](https://github.com/user-attachments/assets/0ce44d9d-5898-48ac-85d3-dd0db661d5e4)

![image](https://github.com/user-attachments/assets/7b656b7f-a16b-4bda-b7b4-79342d0b884d)

![image](https://github.com/user-attachments/assets/61b1f265-f80f-4437-9d9b-4249c44b10f1)

### 由于我们希望关注的是return>0的那部分价格跳跃，考虑筛选出其中的“正跳”，即jump_score>0的部分跳跃，则此时jump_score无需跳跃对齐。

##### 用某天的分钟频数据计算D1,D2,D3，分别取三个方向的值最大的jump和最小的jump，记录其jump时间点前后各59分钟Jump_score以及滑动窗口大小为19的平滑Jump_score图像如下：

D1方向的值越大则越不对称；
D2方向的值越大，则反转性越强，趋势性越弱；
D3方向的值越大，则趋势性越强，反转性越弱。
![image](https://github.com/user-attachments/assets/615a72d8-5fad-4310-8aee-a7bad3fcb370)

##### 用某天的分钟频数据计算D1,D2,D3，分别取三个方向的值最大的jump和最小的jump，记录其jump时间点前后各59分钟Return以及滑动窗口大小为19的平滑Return图像如下：
D1方向的值越大则越不对称；
D2方向的值越大，则反转性越强，趋势性越弱；
D3方向的值越大，则趋势性越强，反转性越弱。
![image](https://github.com/user-attachments/assets/b9f70e47-4ea4-48ba-8a35-090c7f388197)

##### 用某天的分钟频数据计算D1,D2,D3，分别取三个方向的值最大的jump和最小的jump，记录其jump时间点前后各59分钟Return以及滑动窗口大小为19的平滑Price图像如下：
D1方向的值越大则越不对称；
D2方向的值越大，则反转性越强，趋势性越弱；
D3方向的值越大，则趋势性越强，反转性越弱。
![image](https://github.com/user-attachments/assets/bd31317f-1738-4fb7-a913-30967172a912)
![image](https://github.com/user-attachments/assets/760974e8-3f9d-4e1f-8c64-f2da10538cf6)
![image](https://github.com/user-attachments/assets/838050f1-debd-450a-a824-1b46df146491)
