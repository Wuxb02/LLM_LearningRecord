import torch

def softmax(x, dim=1):
    """
    手动实现 Softmax，包含数值稳定性处理 (Log-Sum-Exp 技巧的变体)。
    
    参数:
        x (Tensor): 输入张量
        dim (int): 进行 Softmax 的维度
    """
    max_x,_ = torch.max(x,dim=dim,keepdim=True)
    safe_x = x-max_x

    exp_x = torch.exp(safe_x)
    sum_exp = torch.sum(exp_x,dim=dim,keepdim=True)

    return exp_x/sum_exp

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