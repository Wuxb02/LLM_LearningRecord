import torch

def softmax(x, dim=1):
    """
    手动实现 Softmax，包含数值稳定性处理 (Log-Sum-Exp 技巧的变体)。
    
    参数:
        x (Tensor): 输入张量
        dim (int): 进行 Softmax 的维度
    """
    # 1. 寻找最大值 (keepdim=True 对后续广播非常重要)
    # torch.max 返回 (values, indices)，我们只需要 values
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    
    # 2. 减去最大值 (为了数值稳定性，防止 exp 爆炸)
    # 如果 x 很大，exp(x) 会溢出；但 exp(x - x_max) 最大为 exp(0)=1，非常安全
    x_safe = x - x_max
    
    # 3. 计算指数
    exp_x = torch.exp(x_safe)
    
    # 4. 计算分母（指数之和）
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    
    # 5. 归一化计算概率
    return exp_x / sum_exp_x

def run_tests():
    print(f"{'='*20} 开始测试 {'='*20}")

    # --- 测试用例 1: 基础小数值测试 ---
    print("\n[测试 1] 基础小数值输入:")
    input_small = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    output_small = softmax(input_small, dim=1)
    print(f"输入: {input_small.tolist()}")
    print(f"输出: {output_small.tolist()}")
    # 验证概率和是否为 1
    print(f"概率和: {output_small.sum().item():.4f} (应为 1.0)")


    # --- 测试用例 2: 数值稳定性测试 (大数值) ---
    print("\n[测试 2] 数值稳定性 (输入包含大数值 1000):")
    # 如果不减去最大值，exp(1000) 会变成 inf，导致结果为 nan
    input_large = torch.tensor([[1000.0, 1001.0, 999.0]]) 
    output_large = softmax(input_large, dim=1)
    
    print(f"输入: {input_large.tolist()}")
    print(f"输出: {output_large.tolist()}")
    
    if torch.isnan(output_large).any():
        print("❌ 失败: 结果包含 NaN (数值溢出)")
    else:
        print("✅ 成功: 处理了大数值没有溢出")


    # --- 测试用例 3: 与 PyTorch 官方 API 对比 ---
    print("\n[测试 3] 与 torch.nn.functional.softmax 对比:")
    data = torch.randn(5, 10) # 模拟一个 Batch=5, Class=10 的数据
    
    my_result = softmax(data, dim=1)
    torch_result = torch.nn.functional.softmax(data, dim=1)
    
    # 使用 allclose 比较浮点数是否足够接近
    is_same = torch.allclose(my_result, torch_result, atol=1e-7)
    
    print(f"两个实现是否一致? {'✅ 是' if is_same else '❌ 否'}")
    
    # 展示第一行的差异 (如果有的话，应该是极小的浮点误差)
    diff = (my_result - torch_result).abs().max()
    print(f"最大误差值: {diff.item()}")

if __name__ == "__main__":
    run_tests()