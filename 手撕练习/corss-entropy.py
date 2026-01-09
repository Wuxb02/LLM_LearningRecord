import torch
import torch.nn.functional as F

def manual_cross_entropy(logits, target, reduction='mean'):
    """
    手动实现 CrossEntropyLoss (LogSoftmax + NLLLoss)
    
    参数:
        logits: 模型输出的原始分数 (Batch_Size, Num_Classes)
        target: 真实标签索引 (Batch_Size,)
        reduction: 'mean' (平均值) 或 'sum' (求和) 或 'none' (不归约)
    """
    
    # --- 步骤 1: 实现 LogSoftmax (数值稳定版本) ---
    # LogSoftmax(x_i) = x_i - Log(Sum(exp(x_j)))
    # 为了防止 exp 溢出，使用 log_sum_exp 技巧:
    # log(sum(exp(x))) = m + log(sum(exp(x - m))), 其中 m 是最大值
    
    # 1.1 找到每行的最大值 (保持维度以便广播)
    max_logits, _ = torch.max(logits, dim=1, keepdim=True)
    
    # 1.2 计算稳定的 Log-Sum-Exp
    # 先减去最大值，再 exp，再 sum，再 log，最后加上最大值
    # shape: (Batch_Size, 1)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - max_logits), dim=1, keepdim=True)) + max_logits
    
    # 1.3 计算 Log Probabilities
    # shape: (Batch_Size, Num_Classes)
    log_probs = logits - log_sum_exp
    
    # --- 步骤 2: 实现 NLLLoss (负对数似然) ---
    # 我们需要根据 target 索引，把对应类别的 log_prob 取出来
    # log_probs[i, target[i]]
    
    batch_size = logits.size(0)
    # 使用 advanced indexing 获取对应目标类别的 log_prob
    # range(batch_size) 生成行索引 [0, 1, ...], target 是列索引
    chosen_log_probs = log_probs[range(batch_size), target]
    
    # 取负号 (因为是负对数似然)
    nll_loss = -chosen_log_probs
    
    # --- 步骤 3: Reduction (归约) ---
    if reduction == 'mean':
        return nll_loss.mean()
    elif reduction == 'sum':
        return nll_loss.sum()
    else:
        return nll_loss