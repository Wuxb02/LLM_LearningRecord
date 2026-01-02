import torch
import torch.nn.functional as F
import math

def attention(query, key, value, mask=None, dropout=None):
    """
    手动实现 Scaled Dot-Product Attention.
    
    参数:
        query: 查询张量 (Batch, Seq_len_q, Dim_k)
        key:   键张量   (Batch, Seq_len_k, Dim_k)
        value: 值张量   (Batch, Seq_len_v, Dim_v)
        mask:  掩码张量 (Batch, Seq_len_q, Seq_len_k) 或 (Seq_len_q, Seq_len_k)
               0 表示遮挡 (mask), 1 表示保留 (keep)。
        dropout: dropout 概率 (0.0 表示不使用)
        
    返回:
        output: 加权后的上下文向量 (Batch, Seq_len_q, Dim_v)
        attn_weights: 注意力权重矩阵 (Batch, Seq_len_q, Seq_len_k)
    """
    
    # 获取特征维度 d_k (用于缩放)
    d_k = query.size(-1)
    
    # -------------------------------------------------------
    # 1. 计算原始分数 (Scores) = Q * K^T
    # -------------------------------------------------------
    # key.transpose(-2, -1) 将最后两个维度交换，即 (Batch, Dim_k, Seq_len_k)
    # 结果 shape: (Batch, Seq_len_q, Seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # -------------------------------------------------------
    # 2. 缩放 (Scaling)
    # -------------------------------------------------------
    # 除以 sqrt(d_k) 是为了防止点积结果过大导致 Softmax 梯度消失
    scores = scores / math.sqrt(d_k)
    
    # -------------------------------------------------------
    # 3. 应用掩码 (Masking)
    # -------------------------------------------------------
    if mask is not None:
        # masked_fill: 如果 mask 为 0，则将 score 填充为极小的负数 (-1e9)
        # 这样 softmax 后对应的概率就会趋近于 0
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # -------------------------------------------------------
    # 4. Softmax 归一化 -> 得到注意力权重
    # -------------------------------------------------------
    # 在最后一个维度 (key 的维度) 上进行 softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # (可选) Dropout
    if dropout is not None:
        attn_weights = F.dropout(attn_weights, p=dropout)
    
    # -------------------------------------------------------
    # 5. 加权求和 = Weights * V
    # -------------------------------------------------------
    # (Batch, Seq_len_q, Seq_len_k) * (Batch, Seq_len_v, Dim_v)
    # 注意: Seq_len_k 通常等于 Seq_len_v
    # 结果 shape: (Batch, Seq_len_q, Dim_v)
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights

def run_tests():
    print(f"{'='*20} 开始 Attention 测试 {'='*20}")
    
    # 设定随机种子以保证结果可复现
    torch.manual_seed(42)

    # --- 参数设置 ---
    batch_size = 2
    seq_len = 4
    d_model = 8  # embedding 维度
    
    # 创建 Q, K, V (这里为了简单，它们来自同一个输入，即自注意力)
    x = torch.randn(batch_size, seq_len, d_model)
    Q = x
    K = x
    V = x

    # --- 测试 1: 基础运行 ---
    print("\n[测试 1] 基础输入输出形状:")
    output, weights = attention(Q, K, V)
    print(f"输入 Q 形状: {Q.shape}")
    print(f"输出 Output 形状: {output.shape} (应为 {batch_size}, {seq_len}, {d_model})")
    print(f"权重 Weights 形状: {weights.shape} (应为 {batch_size}, {seq_len}, {seq_len})")


    # --- 测试 2: Mask 机制验证 (最关键) ---
    print("\n[测试 2] Mask 功能验证:")
    # 创建一个 Mask，假设我们要遮挡住第 4 个 token (index 3)
    # Mask 形状: (Batch, Seq_len, Seq_len)
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, 3] = 0 # 将最后一列设为 0，意味着任何 token 都不能关注第 4 个 token
    
    _, weights_masked = attention(Q, K, V, mask=mask)
    
    # 检查第 4 列的权重是否为 0
    last_token_weight = weights_masked[0, 0, 3].item() # Batch 0, Query 0, Key 3
    print(f"被 Mask 的位置权重: {last_token_weight:.10f}")
    
    if last_token_weight < 1e-8:
        print("✅ 成功: Mask 生效，权重趋近于 0")
    else:
        print("❌ 失败: Mask 未生效")


    # --- 测试 3: 与 PyTorch 2.0+ 官方 API 对比 ---
    # PyTorch 2.0 引入了 F.scaled_dot_product_attention (通常也是由 CUDA 优化过的)
    if hasattr(F, 'scaled_dot_product_attention'):
        print("\n[测试 3] 与 torch.nn.functional.scaled_dot_product_attention 对比:")
        
        # 官方 API 也可以接受 mask，但逻辑稍有不同（通常是 bool mask 或 additive mask）
        # 这里我们对比无 mask 的情况以验证数学公式
        
        # 我们的实现
        my_out, _ = attention(Q, K, V)
        
        # 官方实现 (注意：官方版本默认不做 dropout，返回仅仅是 output)
        torch_out = F.scaled_dot_product_attention(Q, K, V)
        
        is_close = torch.allclose(my_out, torch_out, atol=1e-6)
        print(f"与官方实现结果一致? {'✅ 是' if is_close else '❌ 否'}")
        
        diff = (my_out - torch_out).abs().max()
        print(f"最大误差: {diff.item()}")
    else:
        print("\n[跳过测试 3] 你的 PyTorch 版本低于 2.0，没有官方 SDPA 函数。")

if __name__ == "__main__":
    run_tests()