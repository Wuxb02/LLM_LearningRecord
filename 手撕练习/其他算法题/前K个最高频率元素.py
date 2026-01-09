from collections import Counter

def topKFrequent(nums, k):
    # 1. 统计频率
    # 结果示例: {1: 3, 2: 2, 3: 1}
    count = Counter(nums)
    
    # 2. 创建桶
    # 桶的索引表示频率，内容是具有该频率的数字列表
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in count.items():
        buckets[freq].append(num)
    
    # 3. 逆序从桶中提取前 k 高频的元素
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result

# 示例调用
nums = [1, 1, 1, 2, 2,2, 3]
k = 1
print(f"输入: nums = {nums}, k = {k}")
print(f"结果: {topKFrequent(nums, k)}")